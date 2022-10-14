import os
import cv2, torch
import numpy as np
from PIL import Image
from collections import defaultdict
from datasets.utils import read_pfm
from torch.utils.data import Dataset
from torchvision import transforms as T


class MVSDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, levels=3, crop_wh=None):
        assert split in ['train', 'val', 'all'], \
            'split must be either "train", "val" or "all"!'
        if crop_wh is not None:
            assert crop_wh[0] % 32 == 0 and crop_wh[1] % 32 == 0, \
                'crop_wh must both be multiples of 32!'

        self.root_dir = root_dir
        self.split = split
        self.crop_wh = crop_wh
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.n_depths = 192
        self.img_wh = (768, 576)

        self.scale_factors = {}
        self.build_metas()
        self.cal_crop_factors()
        self.define_transforms()

    def cal_crop_factors(self):
        """"calculate crop factors"""
        self.start_w = (self.img_wh[0] - self.crop_wh[0]) // 2
        self.start_h = (self.img_wh[1] - self.crop_wh[1]) // 2
        self.finish_w = self.start_w + self.crop_wh[0]
        self.finish_h = self.start_h + self.crop_wh[1]

    def build_metas(self):
        self.metas = []
        self.ref_views_per_scan = defaultdict(list)
        if self.split == 'train':
            list_txt = os.path.join(self.root_dir, 'training_list.txt')
        elif self.split == 'val':
            list_txt = os.path.join(self.root_dir, 'validation_list.txt')
        else:
            list_txt = os.path.join(self.root_dir, 'all_list.txt')

        with open(list_txt) as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        for scan in self.scans:
            with open(os.path.join(self.root_dir, scan, "cams/pair.txt")) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    self.ref_views_per_scan[scan] += [ref_view]
                    line = f.readline().rstrip().split()
                    n_views_valid = int(line[0]) # valid views
                    if n_views_valid < self.n_views: # skip no enough valid views
                        continue
                    src_views = [int(x) for x in line[1::2]]
                    self.metas += [(scan, -1, ref_view, src_views)]

    def read_cam_file(self, scan, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        if depth_min < 0: depth_min = 0.01
        if scan not in self.scale_factors:
            # use the first cam to determine scale factor
            self.scale_factors[scan] = 100 / depth_min

        depth_min *= self.scale_factors[scan]
        extrinsics[:3, 3] *= self.scale_factors[scan]
        return intrinsics, extrinsics, depth_min

    def read_depth_and_mask(self, scan, filename, depth_min):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth *= self.scale_factors[scan]

        depth_3 = depth[self.start_h:self.finish_h, self.start_w:self.finish_w]
        depth_2 = cv2.resize(depth_3, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        depth_1 = cv2.resize(depth_3, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

        depths = {"stage1": torch.FloatTensor(depth_1),
                  "stage2": torch.FloatTensor(depth_2),
                  "stage3": torch.FloatTensor(depth_3)}

        masks = {"stage1": torch.BoolTensor(depth_1 > depth_min),
                 "stage2": torch.BoolTensor(depth_2 > depth_min),
                 "stage3": torch.BoolTensor(depth_3 > depth_min)}

        depth_max = depth_3.max()
        
        return depths, masks, depth_max

    def define_transforms(self):
        if self.split == 'train': # you can add augmentation here
            self.transform = T.Compose([T.ColorJitter(brightness=0.25,
                                                      contrast=0.5),
                                        T.ToTensor(),
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
        scan, _, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        proj_mats = [] # record proj mats between views
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(
                self.root_dir, f'{scan}/blended_images/{vid:08d}.jpg')
            depth_filename = os.path.join(
                self.root_dir, f'{scan}/rendered_depth_maps/{vid:08d}.pfm')
            proj_mat_filename = os.path.join(
                self.root_dir, scan, f'cams/{vid:08d}_cam.txt')

            img = Image.open(img_filename)
            img = img.crop((self.start_w, self.start_h, self.finish_w, self.finish_h))
            img = self.transform(img)
            imgs += [img]
            
            intrinsics, extrinsics, depth_min = self.read_cam_file(scan, proj_mat_filename)
            intrinsics[0] *= 0.25
            intrinsics[1] *= 0.25
            intrinsics[0, 2] = intrinsics[0, 2] - self.start_w * 0.25
            intrinsics[1, 2] = intrinsics[1, 2] - self.start_h * 0.25
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_mats.append(torch.FloatTensor(proj_mat))

            if i == 0:  # reference view
                depth_ms, mask_ms, depth_max = self.read_depth_and_mask(scan, depth_filename, depth_min)
                if depth_max < depth_min: depth_max = 2 * depth_min
                depth_interval = (depth_max-depth_min) / self.n_depths
                depth_values = torch.arange(depth_min, depth_max, depth_interval)

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