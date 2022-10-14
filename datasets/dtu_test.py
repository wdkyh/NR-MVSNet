import os
import cv2, torch
import numpy as np
from PIL import Image
from datasets.utils import read_pfm
from torch.utils.data import Dataset
from torchvision import transforms as T

class MVSDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, levels=3, img_wh=(1152, 864)):
        assert split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'
        if img_wh is not None:
            assert img_wh[0]%32==0 and img_wh[1]%32==0, \
                'img_wh must both be multiples of 32!'

        self.split = split
        self.levels = levels # FPN levels
        self.img_wh = img_wh
        self.n_views = n_views
        self.root_dir = root_dir
        
        self.build_metas()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        with open(f'lists/dtu/{self.split}.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        for scan in self.scans:
            with open(os.path.join(self.root_dir, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    self.metas += [(scan, ref_view, src_views)]

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
        scan, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_mats = [] # record proj mats between views
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(
                self.root_dir, '{}/images/{:0>8}.jpg'.format(scan, vid)
            )
            proj_mat_filename = os.path.join(
                self.root_dir, '{}/cams/{:0>8}_cam.txt'.format(scan, vid)
            )

            img = Image.open(img_filename)
            img = img.resize(self.img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            intrinsics, extrinsics, depth_min = self.read_cam_file(proj_mat_filename)
            intrinsics[0] *= self.img_wh[0] / 1600 / 4
            intrinsics[1] *= self.img_wh[1] / 1200 / 4
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_mats.append(torch.FloatTensor(proj_mat))

            if i == 0:  # reference view
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
        sample['depth_values'] = depth_values
        sample['filename'] = scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"

        return sample
