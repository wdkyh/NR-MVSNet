import torch
import torch.nn as nn
import torch.nn.functional as F

from .stagenet import *
from .submodule import *

Align_Corners = False


class MVSNet(nn.Module):
    def __init__(self, num_range_samples=[32, 24, 8], num_normal_samples=[16, 8, 0], iterations=[2, 2, 1], depth_intervals_ratio=[4, 2, 1]):
        super(MVSNet, self).__init__()

        self.num_range_samples = num_range_samples
        self.num_normal_samples = num_normal_samples
        self.num_stages = len(num_range_samples)
        self.depth_intervals_ratio = depth_intervals_ratio

        self.stage_infos = {
            "stage1": {"scale": 4.0},
            "stage2": {"scale": 2.0},
            "stage3": {"scale": 1.0}
        }

        self.feature_net = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stages, arch_mode='fpn')
        
        feature_channels = self.feature_net.out_channels
        self.stage_nets = nn.ModuleList([
                        StageNet(stage=i,
                                 iteration=iterations[i],
                                 feat_channels=feature_channels[i],
                                 num_range_samples=self.num_range_samples[i],
                                 num_normal_samples=self.num_normal_samples[i]
                        ) for i in range(self.num_stages)]
        )

    def forward(self, imgs, proj_matrices, depth_values):

        batch, nview, _, img_height, img_width = imgs.shape

        # calc depth interval
        img_min_depth = depth_values[:, 0].view(batch, 1, 1, 1)
        img_max_depth = depth_values[:, -1].view(batch, 1, 1, 1)
        depth_interval = (img_max_depth - img_min_depth) / depth_values.size(1)

        # feature extraction
        features = []
        for nview_idx in range(nview):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature_net(img))
        
        outputs = {}
        depth, cur_depth, uncertainty_map, view_weights = None, None, None, None
        for stage_idx in range(self.num_stages):
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]

            cur_width = img_width // int(stage_scale)
            cur_height = img_height // int(stage_scale)
            
            if depth is not None:
                cur_depth = depth.detach()
                cur_depth = F.interpolate(
                                cur_depth.unsqueeze(1),
                                [cur_height, cur_width],
                                mode='bilinear',
                                align_corners=Align_Corners
                )
                uncertainty_map = F.interpolate(
                                    uncertainty_map.unsqueeze(1),
                                    [cur_height, cur_width],
                                    mode='bilinear',
                                    align_corners=Align_Corners
                )
                view_weights = F.interpolate(
                                    view_weights,
                                    [cur_height, cur_width],
                                    mode="nearest"
                )
            else:
                cur_depth = depth_values

            if stage_idx == 2:
                next_depth_interval_pixel = self.depth_intervals_ratio[stage_idx] * depth_interval
            else:
                next_depth_interval_pixel = self.depth_intervals_ratio[stage_idx + 1] * depth_interval

            output_stage = self.stage_nets[stage_idx](
                            depth = cur_depth,
                            features=features_stage,
                            shape=[batch, cur_height, cur_width],
                            proj_matrices=proj_matrices_stage,
                            depth_interval_pixel=self.depth_intervals_ratio[stage_idx] * depth_interval,
                            next_depth_interval_pixel=next_depth_interval_pixel,
                            uncertainty_map=uncertainty_map,
                            view_weights=view_weights,
                            img_min_depth=img_min_depth, img_max_depth=img_max_depth
            )

            depth = output_stage["depths"][-1]
            view_weights = output_stage["view_weights"]
            uncertainty_map = output_stage["uncertainty_map"]
            outputs["stage{}".format(stage_idx + 1)] = output_stage
            outputs.update(output_stage)
        
        return outputs
            