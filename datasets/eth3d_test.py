import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class MVSDataset(Dataset):
    def __init__(self, root_dir, split='test', n_views=3, levels=3, img_wh=(1920, 1280)):
        assert img_wh[0] % 32==0 and img_wh[1] % 32==0, \
            'img_wh must both be multiples of 32!'
        assert split in ['train', 'test'], \
            'split must be either "train", "test"!'
        self.img_wh = img_wh
        self.root_dir = root_dir
        
        self.split = split
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.build_metas()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        if self.split == "test":
            self.scans = ['botanical_garden', 'boulders', 'bridge', 'door',
                'exhibition_hall', 'lecture_room', 'living_room', 'lounge',
                'observatory', 'old_computer', 'statue', 'terrace_2']

        elif self.split == "train":
            self.scans = ['courtyard', 'delivery_area', 'electro', 'facade',
                    'kicker', 'meadow', 'office', 'pipes', 'playground',
                    'relief', 'relief_2', 'terrace', 'terrains']

        for scan in self.scans:
            with open(os.path.join(self.root_dir, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    self.metas += [(scan, -1, ref_view, src_views)]

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
        if depth_min < 0: depth_min = 1
        depth_max = float(lines[11].split()[-1])
        return intrinsics, extrinsics, depth_min, depth_max

    def define_transforms(self):
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
                self.root_dir, scan, f'images/{vid:08d}.jpg'
            )
            proj_mat_filename = os.path.join(
                self.root_dir, scan, f'cams_1/{vid:08d}_cam.txt'
            )
            img = Image.open(img_filename)
            original_w, original_h = img.size
            img = img.resize(self.img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            intrinsics, extrinsics, depth_min, depth_max = self.read_cam_file(proj_mat_filename)
            intrinsics[0] *= self.img_wh[0] / original_w / 4
            intrinsics[1] *= self.img_wh[1] / original_h / 4
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_mats.append(torch.FloatTensor(proj_mat))

            if i == 0:  # reference view
                depth_interval = (depth_max - depth_min) / 192
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
        sample['depth_values'] = depth_values
        sample['filename'] = scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"

        return sample