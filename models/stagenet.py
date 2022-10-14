import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters import spatial_gradient
from kornia.geometry.depth import depth_to_3d

from .submodule import *
from .refinenet import DepthUpdate

eps = 1e-6


class DepthSampleRange(nn.Module):
    """ Depth sample use inverse depth range"""

    def __init__(self) -> None:
        """Initialize method"""
        super(DepthSampleRange, self).__init__()
    
    def forward(self, depth, num_samples, depth_interval_pixel, uncertainty_map, shape):
        """Forward function for depth sample with inverse range
        
        Args:
            depth: current depth (B, n) or (B, 1, H, W)
            num_samples: number of samples used in range sample  
            depth_interval_pixel: depth interval scale
            uncertainty_map: the uncertainty prob of the depth
            shape: current shape batch, height, width
            min_depth: the min depth of the image
            img_max_depth: the max depth of the image

        Returns:
            depth_sample: depth map by randomization or inverse sample [B, ndepth, H, W]
            min_depth: current min depth map [B, ndepth, H, W]
            max_depth: current max depth map [B, ndepth, H, W]
        """
        # device = depth.device
        # batch, height, width = shape
        # if depth.dim() == 2:
        #     cur_min_depth, cur_max_depth = depth[:, 0], depth[:, -1]
        #     cur_min_depth = cur_min_depth.view(batch, 1, 1, 1).repeat(1, 1, height, width)
        #     cur_max_depth = cur_max_depth.view(batch, 1, 1, 1).repeat(1, 1, height, width)
        #     inverse_min_depth, inverse_max_depth = 1.0 / cur_min_depth, 1.0 / cur_max_depth
            
        #     depth_sample = torch.rand(size=(batch, self.num_samples, height, width),
        #         device=device) + torch.arange(start=0, end=self.num_samples, step=1, device=device).view(
        #         1, self.num_samples, 1, 1).float()
        # else:
        #     cur_min_depth = (depth - self.num_samples / 2 * uncertainty_map * depth_interval_pixel * 1.5)
        #     cur_max_depth = (depth + self.num_samples / 2 * uncertainty_map * depth_interval_pixel * 1.5)

        #     cur_min_depth = cur_min_depth.clamp(min=img_min_depth)
        #     cur_max_depth = cur_max_depth.clamp(max=img_max_depth)
        #     inverse_min_depth, inverse_max_depth = 1.0 / cur_min_depth, 1.0 / cur_max_depth

        #     depth_sample = torch.arange(start=0, end=self.num_samples, 
        #         step=1, device=device).view(1, self.num_samples, 1, 1).float()
            
        # depth_sample = inverse_max_depth + depth_sample / self.num_samples * (inverse_min_depth - inverse_max_depth)
        # depth_sample = 1.0 / depth_sample

        # return depth_sample, cur_min_depth, cur_max_depth

        device = depth.device
        batch, height, width = shape
        if depth.dim() == 2:
            cur_min_depth, cur_max_depth = depth[:, 0], depth[:, -1]
            cur_min_depth = cur_min_depth.view(batch, 1, 1, 1).repeat(1, 1, height, width)
            cur_max_depth = cur_max_depth.view(batch, 1, 1, 1).repeat(1, 1, height, width)
            
            new_interval = (cur_max_depth - cur_min_depth) / (num_samples - 1)
            depth_sample = cur_min_depth + torch.arange(
                    0, num_samples, device=device, requires_grad=False).float().view(1, -1, 1, 1) * new_interval
        else:
            if uncertainty_map is None: 
                uncertainty_map, constant  = 1.0, 1.0
            else:
                constant = 1.5
            new_interval = uncertainty_map * depth_interval_pixel * constant
            cur_min_depth = torch.clamp_min(depth - num_samples / 2 * depth_interval_pixel, 1e-7)
            depth_sample = cur_min_depth + torch.arange(
                    0, num_samples, device=device, requires_grad=False).float().view(1, -1, 1, 1) * new_interval

            # cur_min_depth = (depth - num_samples / 2 * uncertainty_map * depth_interval_pixel * constant)
            # cur_max_depth = (depth + num_samples / 2 * uncertainty_map * depth_interval_pixel * constant)

            # cur_min_depth = cur_min_depth.clamp(min=img_min_depth)
            # cur_max_depth = cur_max_depth.clamp(max=img_max_depth)

            # if (cur_min_depth >= cur_max_depth).any():
            #     print("min_depth > max_depth")

            # new_interval = (cur_max_depth - cur_min_depth) / (num_samples - 1)

            # depth_sample = cur_min_depth + torch.arange(
            #         0, num_samples, device=device, requires_grad=False).float().view(1, -1, 1, 1) * new_interval
            
        return depth_sample


def get_normalsample_grid(batch, height, width, num_samples, offset, device, dilation=2):
    """Compute the offset for normal depth sample
    Args:
        batch: batch size
        height: grid height
        width: grid width
        num_samples: number of the normal sample
        offset: grid offset
        device: device on which to place tensor
        dilation: original offset dilation
    Returns:
        generated grid: in the shape of [batch, num_samples*height, width, 2]
    """

    if num_samples == 4:  # if 4 neighbors to be sampled
        original_offset = [[-dilation, 0], [0, -dilation], [0, dilation], [dilation, 0]]
    elif num_samples == 8:  # if 8 neighbors to be sampled
        original_offset = [
            [-5 * dilation, -0], [-dilation, 0], [0, -5 * dilation],
            [0, -dilation], [0, dilation],
            [0, 5 * dilation], [dilation, 0], [5 * dilation, 0]
        ]
    elif num_samples == 12:  # if 12 neighbors to be sampled
        original_offset = [
            [-5 * dilation, -0], [-3 * dilation, -0], [-dilation, 0],
            [0, -5 * dilation], [0, -3 * dilation], [0, -dilation],
            [0, dilation], [0, 3 * dilation], [0, 5 * dilation],
            [dilation, 0], [3 * dilation, 0], [5 * dilation, 0]
        ]
    elif num_samples == 16:  # if 16 neighbors to be sampled in propagation
                original_offset = [
                    [-dilation, -dilation],
                    [-dilation, 0],
                    [-dilation, dilation],
                    [0, -dilation],
                    [0, dilation],
                    [dilation, -dilation],
                    [dilation, 0],
                    [dilation, dilation],
                ]
                for i in range(len(original_offset)):
                    offset_x, offset_y = original_offset[i]
                    original_offset.append([2 * offset_x, 2 * offset_y])
    else:
        raise NotImplementedError

    with torch.no_grad():
        y_grid, x_grid = torch.meshgrid(
            [torch.arange(0, height, dtype=torch.float32, device=device),
             torch.arange(0, width, dtype=torch.float32, device=device),
            ]
        )
        y_grid, x_grid = y_grid.contiguous(), x_grid.contiguous()
        y_grid, x_grid = y_grid.view(height * width), x_grid.view(height * width)
        
        xy = torch.stack((x_grid, y_grid))  # [2, height * width]
        xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [batch, 2, height * width]

    xy_list = []
    for i in range(len(original_offset)):
        original_offset_y, original_offset_x = original_offset[i]

        offset_x_tensor = original_offset_x + offset[:, 2 * i, :].unsqueeze(1)
        offset_y_tensor = original_offset_y + offset[:, 2 * i + 1, :].unsqueeze(1)

        xy_list.append((xy + torch.cat((offset_x_tensor, offset_y_tensor), dim=1)).unsqueeze(2))

    xy = torch.cat(xy_list, dim=2)  # [batch, 2, num_samples, height * width]

    del xy_list, x_grid, y_grid

    x_normalized = xy[:, 0, :, :] / ((width - 1) / 2) - 1
    y_normalized = xy[:, 1, :, :] / ((height - 1) / 2) - 1
    del xy
    grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, num_samples, height * width, 2]
    del x_normalized, y_normalized
    grid = grid.view(batch, num_samples * height, width, 2)

    return grid


class DepthSampleNormal(nn.Module):
    """ Depth Sample use neighbors normal implementation"""

    def __init__(self) -> None:
        """Initialize method"""
        super(DepthSampleNormal, self).__init__()
    
    def forward(self, depth, num_neigs, intrinsics, grid, min_depth=None, max_depth=None):
        """Forward method of depth sample with normal
        
        Args:
            depth: predicted depth map, in shape of [batch, 1, height, width]
            num_neigs: number of neighbors to be sampled
            intrinsics: carama intrinsics, in shape of [batch, 3, 3]
            grid: 2D grid for bilinear gridding, in shape of [batch, num_neigs*height, wigth, 2]
            min_depth: minimum virtual depth, in shape of [batch, 1, height, width]
            max_depth: maximum virtual depth, in shape of [batch, 1, height, width]

        Returns:
            normal sample detph: sorted depth map [batch, num_neigs, height, width]
        """
        batch, _, height, width = depth.shape

        # compute the 3d points from depth
        xyz = depth_to_3d(depth, intrinsics)

        # compute normals
        gradients = spatial_gradient(xyz)
        a, b = gradients[:, :, 0], gradients[:, :, 1]
        normals = torch.cross(a, b, dim=1)
        normals = F.normalize(normals, dim=1, p=2)

        # extract points patch
        points_patch = F.grid_sample(xyz, grid, mode="bilinear", padding_mode="border")
        points_patch = points_patch.view(batch, 3, num_neigs, height, width)

        # extract normals patch
        normals_patch = F.grid_sample(normals, grid, mode="bilinear", padding_mode="border")
        normals_patch = normals_patch.view(batch, 3, num_neigs, height, width)

        # point project to surface
        xyz = xyz.unsqueeze(2)  # [batch, 3, num_neigs, height, width]
        val = torch.sum(normals_patch * (points_patch - xyz), dim=1) / (torch.sum(normals_patch ** 2, dim=1) + eps)
        proj_xyz = xyz + val.unsqueeze(1) * normals_patch

        # compute depth
        proj_xyz = proj_xyz.view(batch, 3, -1)
        depth_sample = torch.matmul(intrinsics, proj_xyz)[:, -1, :]
        depth_sample = depth_sample.view(batch, num_neigs, height, width)

        if min_depth is not None:
            min_depth = min_depth.repeat(1, num_neigs, 1, 1)
            depth_sample_clamp = torch.where(depth_sample < min_depth, min_depth, depth_sample)
        if max_depth is not None:
            max_depth = max_depth.repeat(1, num_neigs, 1, 1)
            depth_sample_clamp = torch.where(depth_sample > max_depth, max_depth, depth_sample)
        
        return depth_sample_clamp


class PixelwiseNet(nn.Module):
    """Pixelwise Net: A simple pixel-wise view weight network, composed of 1x1x1 convolution layers
    and sigmoid nonlinearities, takes the initial set of similarities to output a number between 0 and 1 per
    pixel as estimated pixel-wise view weight.
    1. The similarity is calculated by ref_feature and other source_features warped by differentiable_warping
    2. The learned pixel-wise view weight is estimated in the first iteration of Patchmatch and kept fixed in the
    matching cost computation.
    """

    def __init__(self, in_channels: int) -> None:
        """Initialize method
        Args:
            in_channels: the feature channels of input
        """
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=in_channels, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        """Forward method for PixelwiseNet
        Args:
            x1: pixel-wise view weight, [B, in_channels, Ndepth, H, W]
        """
        # [B, Ndepth, H, W]
        x1 = self.conv2(self.conv1(self.conv0(x1))).squeeze(1)

        output = self.output(x1)
        del x1
        # [B,H,W]
        output = torch.max(output, dim=1)[0]

        return output.unsqueeze(1)


class DepthPredNet(nn.Module):
    def __init__(self):
        super(DepthPredNet, self).__init__()

    def forward(self, features, proj_matrices, depth_samples, cost_reg_net, uncertainty_net, pixelwise_net, view_weights):
        
        batch, feat_channel, height, width = features[0].shape
        dtype, device = features[0].dtype, features[0].device

        proj_matrices = torch.unbind(proj_matrices, 1)
        num_samples = depth_samples.shape[1]

        # step 1. feature and matrices extraction
        ref_features, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        pixelwise_weight_sum = 1e-5 * torch.ones((batch, 1, 1, height, width), dtype=dtype, device=device)
        volume_sum = torch.zeros((batch, feat_channel, num_samples, height, width), dtype=dtype, device=device)

        if view_weights is None:
            view_weights_list = []
            for src_fea, src_proj in zip(src_features, src_projs):
                #warpped features
                src_proj_new = src_proj[:, 0].clone()
                src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
                ref_proj_new = ref_proj[:, 0].clone()
                ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
                warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_samples)
                warped_volume = warped_volume * ref_features.unsqueeze(2)

                # calc pixel-wise view weight
                view_weight = pixelwise_net(warped_volume)
                view_weights_list.append(view_weight)

                if self.training:
                    volume_sum = volume_sum + warped_volume * view_weight.unsqueeze(1)
                    pixelwise_weight_sum = pixelwise_weight_sum + view_weight.unsqueeze(1)
                else:
                    volume_sum += warped_volume * view_weight.unsqueeze(1)
                    pixelwise_weight_sum += view_weight.unsqueeze(1)
                
            view_weights = torch.cat(view_weights_list, dim=1)  # [B, nview, H, W]

            # aggregated matching cost across all the source views
            volume_aggregated = volume_sum.div_(pixelwise_weight_sum)
        else:
            idx = 0
            for src_fea, src_proj in zip(src_features, src_projs):
                #warpped features
                src_proj_new = src_proj[:, 0].clone()
                src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
                ref_proj_new = ref_proj[:, 0].clone()
                ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
                warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_samples)
                warped_volume = warped_volume * ref_features.unsqueeze(2)

                view_weight = view_weights[:, idx].unsqueeze(1)
                idx += 1

                if self.training:
                    volume_sum = volume_sum + warped_volume * view_weight.unsqueeze(1)
                    pixelwise_weight_sum = pixelwise_weight_sum + view_weight.unsqueeze(1)
                else:
                    volume_sum += warped_volume * view_weight.unsqueeze(1)
                    pixelwise_weight_sum += view_weight.unsqueeze(1)

            # aggregated matching cost across all the source views
            volume_aggregated = volume_sum.div_(pixelwise_weight_sum)

        # step 3. cost volume regularization
        cost_reg = cost_reg_net(volume_aggregated)
        prob_volume_pre = cost_reg.squeeze(1)

        # step 4. predict depth
        prob_volume = F.softmax(prob_volume_pre, dim=1)  # (B, D, H, W)
        depth = depth_regression(prob_volume, depth_values=depth_samples)

        # step 5. learning uncertainty map through uncertainty awareness module
        uncertainty_map=uncertainty_net(prob_volume.detach()).squeeze(1) if uncertainty_net is not None else None

        # depth: [B, H, W], uncertainty_map [B, H, W], prob_volume [B, D, H, W], volume_aggregated [B, C, D, H, W]
        return depth, uncertainty_map, view_weights.detach(), prob_volume, volume_aggregated


class StageNet(nn.Module):
    def __init__(self, stage, iteration, feat_channels, num_range_samples, num_normal_samples):
        super(StageNet, self).__init__()

        self.stage = stage
        self.iteration = iteration
        self.num_range_samples = num_range_samples
        self.num_normal_samples = num_normal_samples
        self.num_samples = num_range_samples + num_normal_samples

        # depth range sample
        self.depth_sample_range_net = DepthSampleRange()

        # depth normal sample
        # if num_normal_samples > 0:            
        #     self.dilation = 2
        #     self.depth_smaple_normal_net = DepthSampleNormal()

        # depth normal sample
        if num_normal_samples > 0:            
            # adaptive normal sample grid offset generate
            self.dilation = 2
            self.adaptive_grid_offset_net = nn.Conv2d(
                in_channels=feat_channels,
                out_channels=2 * num_normal_samples,
                kernel_size=3, stride=1,
                padding=self.dilation, dilation=self.dilation,
                bias=True
            )
            nn.init.constant_(self.adaptive_grid_offset_net.weight, 0.0)
            nn.init.constant_(self.adaptive_grid_offset_net.bias, 0.0)

            self.depth_smaple_normal_net = DepthSampleNormal()

        # cost volume regularization
        self.cost_reg_net = CostRegNet(in_channels=feat_channels, base_channels=8)

        # pixel-wise weight net
        if self.stage == 0:
            self.pixelwise_net = PixelwiseNet(in_channels=feat_channels)
        else:
            self.pixelwise_net = None

        # uncertainty awareness of depth predict module to adaptive sample depth range
        if stage == 2 and iteration <= 1:
            self.uncertainty_net = None
        else:
            self.uncertainty_net = REM(in_channels=self.num_samples)

        # dpeht predict moudule
        self.depth_pred_net = DepthPredNet()

        # stage one depth update
        if stage == 0 and num_normal_samples > 0:
            self.depth_update_net = DepthUpdate(feat_channels, n_query_channels=16)
    
    def forward(self,
                depth, features,
                shape, proj_matrices,
                depth_interval_pixel,
                next_depth_interval_pixel,
                uncertainty_map, view_weights,
                img_min_depth, img_max_depth):

        device = depth.device
        batch, height, width = shape
        ref_feat = features[0]

        depths = []
        cur_depth, cur_uncertainty_map = depth, uncertainty_map

        img_min_depth = img_min_depth.repeat(1, 1, height, width)
        img_max_depth = img_max_depth.repeat(1, 1, height, width)

        # at first stage, depth init
        if self.stage == 0 and self.num_normal_samples > 0:
            depth_samples = self.depth_sample_range_net(
                    depth=cur_depth,
                    num_samples=self.num_samples,
                    depth_interval_pixel=depth_interval_pixel,
                    uncertainty_map=cur_uncertainty_map,
                    shape=shape
            )
            depth_pred_net_out = self.depth_pred_net(
                            features=features,
                            proj_matrices=proj_matrices,
                            depth_samples=depth_samples,
                            cost_reg_net=self.cost_reg_net,
                            uncertainty_net=None,
                            pixelwise_net=self.pixelwise_net,
                            view_weights=view_weights
            )
            init_depth, _, view_weights, prob_volume, cost_volume = depth_pred_net_out
            depths.append(init_depth)

            update_depth = self.depth_update_net(
                ref_feat.detach(),
                cost_volume.detach(),
                prob_volume.detach(),
                init_depth.detach().unsqueeze(1),
                next_depth_interval_pixel,
                self.num_samples
            )
            min_depth = img_min_depth.squeeze(1)
            max_depth = img_max_depth.squeeze(1)
            update_depth = torch.where(update_depth < min_depth, min_depth, update_depth)
            update_depth = torch.where(update_depth > max_depth, max_depth, update_depth)
            depths.append(update_depth)

            cur_depth = depths[-1].detach().unsqueeze(1)
        
        # calc the normal sample grid, each iteration the grid is same
        if self.num_normal_samples > 0:
            normal_sample_grid_offset = self.adaptive_grid_offset_net(ref_feat.detach())
            normal_sample_grid_offset = normal_sample_grid_offset.view(batch, 2 * self.num_normal_samples, height * width)
            normal_sample_grid = get_normalsample_grid(
                        batch, height, width,
                        num_samples=self.num_normal_samples,
                        offset=normal_sample_grid_offset,
                        dilation=self.dilation,
                        device=device
            )

        for i in range(self.iteration):
            depth_range_samples = self.depth_sample_range_net(
                        depth=cur_depth,
                        num_samples=self.num_range_samples,
                        depth_interval_pixel=depth_interval_pixel,
                        uncertainty_map=cur_uncertainty_map,
                        shape=shape
            )
            depth_samples = depth_range_samples

            if self.num_normal_samples > 0:
                depth_normal_samples = self.depth_smaple_normal_net(
                            depth=cur_depth,
                            num_neigs=self.num_normal_samples,
                            intrinsics=proj_matrices[:, 0, 1, :3, :3],
                            grid=normal_sample_grid,
                            min_depth=img_min_depth,
                            max_depth=img_max_depth
                )
                depth_samples = torch.cat([depth_samples, depth_normal_samples], dim=1)
                depth_samples, _ = torch.sort(depth_samples, dim=1)

            if self.stage == 2 and i == (self.iteration - 1):
                uncertainty_net = None
            else:
                uncertainty_net = self.uncertainty_net

            depth_pred_net_out = self.depth_pred_net(
                                    features=features, 
                                    proj_matrices=proj_matrices,
                                    depth_samples=depth_samples,
                                    cost_reg_net=self.cost_reg_net,
                                    uncertainty_net=uncertainty_net,
                                    pixelwise_net=self.pixelwise_net,
                                    view_weights=view_weights
            )
            depth, uncertainty_map, view_weights, prob_volume, cost_volume = depth_pred_net_out
            depths.append(depth)
            cur_depth = depth.detach().unsqueeze(1)
            cur_uncertainty_map = uncertainty_map.unsqueeze(1) if uncertainty_map is not None else None

        # In first stage, depth refine 
        # if self.stage == 0:
        #     update_depth = self.depth_update_net(
        #         ref_feat.detach(),
        #         cost_volume.detach(),
        #         prob_volume.detach(),
        #         depth.detach().unsqueeze(1),
        #         next_depth_interval_pixel,
        #         self.num_samples
        #     )
        #     min_depth = img_min_depth.squeeze(1)
        #     max_depth = img_max_depth.squeeze(1)
        #     update_depth = torch.where(update_depth < min_depth, min_depth, update_depth)
        #     update_depth = torch.where(update_depth > max_depth, max_depth, update_depth)
        #     depths.append(update_depth)

        # confidence
        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), 
                            pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume,
                            depth_values=torch.arange(self.num_samples, device=prob_volume.device, dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=self.num_samples - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        return {"depth": depths[-1], "depths": depths,
                "view_weights": view_weights,
                "uncertainty_map": uncertainty_map,
                "photometric_confidence": photometric_confidence}



        



        
