import torch
import numpy as np
from typing import Tuple


def dihedral_transform(grid: torch.Tensor, transform_id: int) -> torch.Tensor:
    """apply dihedral transform (rotation + mirror)
    transform_id: 0-7 (0=id, 1-3=rot90,180,270, 4-7=mirror+rot)
    """
    h, w = grid.shape[-2:]
    grid_2d = grid.view(-1, h, w)

    if transform_id == 0:  # identity
        return grid
    elif transform_id == 1:  # rot90
        return torch.rot90(grid_2d, k=1, dims=[-2, -1]).view(grid.shape)
    elif transform_id == 2:  # rot180
        return torch.rot90(grid_2d, k=2, dims=[-2, -1]).view(grid.shape)
    elif transform_id == 3:  # rot270
        return torch.rot90(grid_2d, k=3, dims=[-2, -1]).view(grid.shape)
    elif transform_id == 4:  # mirror
        return torch.flip(grid_2d, dims=[-1]).view(grid.shape)
    elif transform_id == 5:  # mirror + rot90
        return torch.rot90(torch.flip(grid_2d, dims=[-1]), k=1, dims=[-2, -1]).view(
            grid.shape
        )
    elif transform_id == 6:  # mirror + rot180
        return torch.rot90(torch.flip(grid_2d, dims=[-1]), k=2, dims=[-2, -1]).view(
            grid.shape
        )
    elif transform_id == 7:  # mirror + rot270
        return torch.rot90(torch.flip(grid_2d, dims=[-1]), k=3, dims=[-2, -1]).view(
            grid.shape
        )
    else:
        return grid


def color_permutation(grid: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """apply color permutation (0-9 colors)"""
    return perm[grid]


def small_translation(grid: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """apply small translation (±1-2 pixels)"""
    h, w = grid.shape[-2:]
    grid_2d = grid.view(-1, h, w)

    # pad with zeros
    padded = torch.nn.functional.pad(grid_2d, (2, 2, 2, 2), mode="constant", value=0)

    # crop with offset
    translated = padded[:, 2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]
    return translated.view(grid.shape)


def augment_pair(
    inp: torch.Tensor,
    out: torch.Tensor,
    apply_dihedral: bool = True,
    apply_color_perm: bool = True,
    apply_translation: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """augment input/output pair consistently"""
    # reshape to 2d for augmentation
    h = w = int(np.sqrt(inp.shape[-1]))
    inp_2d = inp.view(-1, h, w)
    out_2d = out.view(-1, h, w)

    if apply_dihedral:
        transform_id = torch.randint(0, 8, (1,)).item()
        inp_2d = dihedral_transform(inp_2d, transform_id)
        out_2d = dihedral_transform(out_2d, transform_id)

    if apply_color_perm:
        perm = torch.randperm(10)
        inp_2d = color_permutation(inp_2d, perm)
        out_2d = color_permutation(out_2d, perm)

    if apply_translation:
        dx = torch.randint(-2, 3, (1,)).item()
        dy = torch.randint(-2, 3, (1,)).item()
        inp_2d = small_translation(inp_2d, dx, dy)
        out_2d = small_translation(out_2d, dx, dy)

    # flatten back
    inp_aug = inp_2d.view(inp.shape)
    out_aug = out_2d.view(out.shape)

    return inp_aug, out_aug
