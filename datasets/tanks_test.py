import os
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms as T

class MVSDataset(Dataset):
    def __init__(self, root_dir, split='intermediate', n_views=3, levels=3, img_wh=(1152, 864)):
        assert img_wh[0] % 32==0 and img_wh[1] % 32==0, \
            'img_wh must both be multiples of 32!'
        assert split in ['intermediate', 'advanced'], \
            'split must be either "intermediate", "advanced"!'
        self.img_wh = img_wh
        self.root_dir = root_dir
        
        self.split = split
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.build_metas()
        self.define_transforms()

    def build_metas(self):
        self.metas = []
        if self.split == 'intermediate':
            self.scans = ['Family', 'Francis', 'Horse', 'Lighthouse',
                          'M60', 'Panther', 'Playground', 'Train']
            self.image_sizes = {'Family': (1920, 1080),
                                'Francis': (1920, 1080),
                                'Horse': (1920, 1080),
                                'Lighthouse': (2048, 1080),
                                'M60': (2048, 1080),
                                'Panther': (2048, 1080),
                                'Playground': (1920, 1080),
                                'Train': (1920, 1080)}
            self.depth_interval = {'Family': 2.5e-3,
                                   'Francis': 1e-2,
                                   'Horse': 1.5e-3,
                                   'Lighthouse': 1.5e-2,
                                   'M60': 5e-3,
                                   'Panther': 5e-3,
                                   'Playground': 7e-3,
                                   'Train': 5e-3} # depth interval for each scan (hand tuned)
        elif self.split == 'advanced':
            self.scans = ['Auditorium', 'Ballroom', 'Courtroom',
                          'Museum', 'Palace', 'Temple']
            self.image_sizes = {'Auditorium': (1920, 1080),
                                'Ballroom': (1920, 1080),
                                'Courtroom': (1920, 1080),
                                'Museum': (1920, 1080),
                                'Palace': (1920, 1080),
                                'Temple': (1920, 1080)}
            self.depth_interval = {'Auditorium': 3e-2,
                                   'Ballroom': 2e-2,
                                   'Courtroom': 2e-2,
                                   'Museum': 2e-2,
                                   'Palace': 1e-2,
                                   'Temple': 1e-2} # depth interval for each scan (hand tuned)
        self.ref_views_per_scan = defaultdict(list)

        for scan in self.scans:
            with open(os.path.join(self.root_dir, self.split, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) <= 0: continue
                    self.metas += [(scan, -1, ref_view, src_views)]
                    self.ref_views_per_scan[scan] += [ref_view]


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
                self.root_dir, self.split, scan, f'images/{vid:08d}.jpg'
            )
            proj_mat_filename = os.path.join(
                self.root_dir, self.split, scan, f'cams_shortrange/{vid:08d}_cam.txt'
            )
            img = Image.open(img_filename)
            img = img.resize(self.img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            img_w, img_h = self.image_sizes[scan]
            intrinsics, extrinsics, depth_min = self.read_cam_file(proj_mat_filename)
            intrinsics[0] *= self.img_wh[0] / img_w / 4
            intrinsics[1] *= self.img_wh[1] / img_h / 4
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_mats.append(torch.FloatTensor(proj_mat))

            if i == 0:  # reference view
                depth_max = self.depth_interval[scan] * 192 + depth_min
                depth_values = torch.arange(depth_min, depth_max, self.depth_interval[scan])
                
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