import torch

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.color_utils import colormap
from easyvolcap.utils.ray_utils import create_meshgrid
from easyvolcap.utils.math_utils import affine_inverse, torch_inverse_3x3


def depth2xyz(depth: torch.Tensor, ixts: torch.Tensor, exts: torch.Tensor, correct_pix: bool = True):
    # depth: B, S, 1, H, W or B, S, H, W
    # exts: B, S, 4, 4
    # ixts: B, S, 3, 3

    # Deal with the depth dimension
    if depth.ndim > 4:
        depth = depth[..., 0, :, :]  # remove the channel dimension for shape preparation

    # Prepare shapes
    B, S, H, W = depth.shape
    depth = depth.reshape((-1, ) + depth.shape[-2:])  # (B * S, H, W)
    ixts = ixts.reshape((-1, ) + ixts.shape[-2:])  # (B * S, 3, 3)
    # The input exts is world to camera by default
    c2ws = affine_inverse(exts).reshape((-1, ) + exts.shape[-2:])  # (B * S, 4, 4)

    # Create the meshgrid for source images
    ref_grid = create_meshgrid(H, W, device=depth.device, correct_pix=correct_pix, dtype=depth.dtype).flip(-1)  # (H, W, 2), in ij ordering
    ref_grid = torch.cat([ref_grid, torch.ones_like(ref_grid[..., :1])], dim=-1)  # (H, W, 3)
    ref_grid = ref_grid[None].expand(B * S, -1, -1, -1).permute(0, 3, 1, 2).reshape(B * S, 3, -1)  # (B * S, 3, H * W)

    # Compute the xyz coordinates in the camera space
    cam_xyz = torch_inverse_3x3(ixts) @ ref_grid  # (B * S, 3, H * W)
    cam_xyz = cam_xyz * depth.reshape(B * S, 1, -1)  # (B * S, 3, H * W)

    # Transform the xyz coordinates to the world space
    world_xyz = c2ws[..., :3, :3] @ cam_xyz + c2ws[..., :3, 3:]  # (B * S, 3, H * W)
    world_xyz = world_xyz.reshape(B, S, 3, H, W)  # (B, S, 3, H, W)

    return world_xyz
