import os
import cv2, torch
import numpy as np
from PIL import Image
from datasets.utils import read_pfm
from torch.utils.data import Dataset
from torchvision import transforms as T

class MVSDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, levels=3, depth_interval=2.65):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val", "test" '
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.depth_interval = depth_interval

        self.build_metas()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        with open(f'lists/dtu/{self.split}.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        light_idxs = range(7)
        pair_file = "Cameras/pair.txt"
        for scan in self.scans:
            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    for light_idx in light_idxs:
                        self.metas += [(scan, light_idx, ref_view, src_views)]

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        return intrinsics, extrinsics, depth_min

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) # (1200, 1600)
        depth = cv2.resize(depth, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
        depth_3 = depth[44:556, 80:720] # (512, 640)
        depth_2 = cv2.resize(depth_3, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)
        depth_1 = cv2.resize(depth_2, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)

        depths = {"stage3": torch.FloatTensor(depth_3),
                  "stage2": torch.FloatTensor(depth_2),
                  "stage1": torch.FloatTensor(depth_1)}
        
        return depths

    def read_mask(self, filename):
        mask = cv2.imread(filename, 0) # (1200, 1600)
        mask = cv2.resize(
            mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST
        ) # (600, 800)

        mask_3 = mask[44:556, 80:720] # (512, 640)
        mask_2 = cv2.resize(mask_3, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)
        mask_1 = cv2.resize(mask_2, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)

        masks = {"stage3": torch.BoolTensor(mask_3),
                 "stage2": torch.BoolTensor(mask_2),
                 "stage1": torch.BoolTensor(mask_1)}

        return masks

    def define_transforms(self):
        if self.split == 'train': # you can add augmentation here
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
        else:
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        sample = {}
        scan, light_idx, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_mats = [] # record proj mats between views
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(
                self.root_dir, f'Rectified/{scan}/rect_{vid+1:03d}_{light_idx}_r5000.png'
            )
            mask_filename = os.path.join(
                self.root_dir, f'Depths/{scan}/depth_visual_{vid:04d}.png'
            )
            depth_filename = os.path.join(
                self.root_dir, f'Depths/{scan}/depth_map_{vid:04d}.pfm'
            )
            proj_mat_filename = os.path.join(
                self.root_dir, f'Cameras/train/{vid:08d}_cam.txt'
            )
            
            img = Image.open(img_filename)
            img = self.transform(img)
            imgs += [img]

            intrinsics, extrinsics, depth_min = self.read_cam_file(proj_mat_filename)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_mats.append(torch.FloatTensor(proj_mat))

            if i == 0:  # reference view
                mask_ms = self.read_mask(mask_filename)
                depth_ms = self.read_depth(depth_filename)

                depth_max = 2.65 * 192 + depth_min
                depth_values = torch.arange(depth_min, depth_max, 2.65)

        imgs = torch.stack(imgs) # (V, 3, H, W)

        # ms proj_mats
        proj_mats_stage1 = torch.stack(proj_mats)
        proj_mats_stage2 = proj_mats_stage1.clone()
        proj_mats_stage2[:, 1, :2, :] = proj_mats_stage2[:, 1, :2, :] * 2
        proj_mats_stage3 = proj_mats_stage1.clone()
        proj_mats_stage3[:, 1, :2, :] = proj_mats_stage3[:, 1, :2, :] * 4
        proj_mats_ms = {
            "stage1": proj_mats_stage1,
            "stage2": proj_mats_stage2,
            "stage3": proj_mats_stage3
        }

        sample['imgs'] = imgs
        sample['proj_mats'] = proj_mats_ms
        sample['depth'] = depth_ms
        sample['mask'] = mask_ms
        sample['depth_values'] = depth_values

        return sample