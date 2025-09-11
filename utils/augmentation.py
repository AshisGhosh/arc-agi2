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
        return torch.rot90(grid_2d, k=1, dims=[-2, -1]).reshape(grid.shape)
    elif transform_id == 2:  # rot180
        return torch.rot90(grid_2d, k=2, dims=[-2, -1]).reshape(grid.shape)
    elif transform_id == 3:  # rot270
        return torch.rot90(grid_2d, k=3, dims=[-2, -1]).reshape(grid.shape)
    elif transform_id == 4:  # mirror
        return torch.flip(grid_2d, dims=[-1]).reshape(grid.shape)
    elif transform_id == 5:  # mirror + rot90
        return torch.rot90(torch.flip(grid_2d, dims=[-1]), k=1, dims=[-2, -1]).reshape(
            grid.shape
        )
    elif transform_id == 6:  # mirror + rot180
        return torch.rot90(
            torch.flip(grid_2d, dims=[-2, -1]), k=2, dims=[-2, -1]
        ).reshape(grid.shape)
    elif transform_id == 7:  # mirror + rot270
        return torch.rot90(torch.flip(grid_2d, dims=[-1]), k=3, dims=[-2, -1]).reshape(
            grid.shape
        )
    else:
        return grid


def color_permutation(grid: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """apply color permutation (0-9 colors)"""
    # ensure perm is on the same device as grid
    if perm.device != grid.device:
        perm = perm.to(grid.device)
    return perm[grid]


def small_translation(grid: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """apply small translation (±1-2 pixels)"""
    h, w = grid.shape[-2:]
    grid_2d = grid.view(-1, h, w)

    # pad with zeros
    padded = torch.nn.functional.pad(grid_2d, (2, 2, 2, 2), mode="constant", value=0)

    # crop with offset
    translated = padded[:, 2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]
    return translated.reshape(grid.shape)


def augment_pair(
    inp: torch.Tensor,
    out: torch.Tensor,
    apply_dihedral: bool = True,
    apply_color_perm: bool = True,
    apply_translation: bool = False,  # disabled for now
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
        perm = torch.randperm(10, device=inp.device)
        inp_2d = color_permutation(inp_2d, perm)
        out_2d = color_permutation(out_2d, perm)

    if apply_translation:
        dx = torch.randint(-2, 3, (1,)).item()
        dy = torch.randint(-2, 3, (1,)).item()
        inp_2d = small_translation(inp_2d, dx, dy)
        out_2d = small_translation(out_2d, dx, dy)

    # flatten back
    inp_aug = inp_2d.reshape(inp.shape)
    out_aug = out_2d.reshape(out.shape)

    # post-augmentation asserts to ensure values stay in [0..9]
    assert (
        inp_aug.min() >= 0 and inp_aug.max() <= 9
    ), f"input values {inp_aug.min()}-{inp_aug.max()}, expected [0..9]"
    assert (
        out_aug.min() >= 0 and out_aug.max() <= 9
    ), f"output values {out_aug.min()}-{out_aug.max()}, expected [0..9]"

    return inp_aug, out_aug


def sample_aug_and_inverse(grid: torch.Tensor, num_augs: int = 8) -> list:
    """generate augmentation samples and their inverse functions

    returns: list of (augmented_grid, inverse_function) tuples
    """
    samples = []
    h = w = int(np.sqrt(grid.shape[-1]))
    grid_2d = grid.view(-1, h, w)

    # dihedral transforms (8 total: identity + 3 rotations + 4 mirror+rot combinations)
    dihedral_count = min(8, num_augs)
    for transform_id in range(dihedral_count):
        aug_grid = dihedral_transform(grid_2d, transform_id)
        samples.append((aug_grid.reshape(grid.shape), transform_id, "dihedral"))

    # color permutations (add remaining variations if needed)
    remaining_augs = num_augs - len(samples)
    for _ in range(remaining_augs):
        perm = torch.randperm(10, device=grid.device)
        aug_grid = color_permutation(grid_2d, perm)
        samples.append((aug_grid.reshape(grid.shape), perm, "color"))

    return samples


def invert_augmentation(
    grid: torch.Tensor, aug_info: tuple, aug_type: str
) -> torch.Tensor:
    """invert an augmentation to get back to original space

    args:
        grid: augmented grid (B, L) or (L,)
        aug_info: augmentation info from sample_aug_and_inverse
        aug_type: type of augmentation ("dihedral" or "color")
    """
    h = w = int(np.sqrt(grid.shape[-1]))
    grid_2d = grid.view(-1, h, w)

    if aug_type == "dihedral":
        transform_id = aug_info
        # invert dihedral transform
        if transform_id == 0:  # identity
            return grid
        elif transform_id == 1:  # rot90 -> rot270
            return torch.rot90(grid_2d, k=3, dims=[-2, -1]).reshape(grid.shape)
        elif transform_id == 2:  # rot180 -> rot180
            return torch.rot90(grid_2d, k=2, dims=[-2, -1]).reshape(grid.shape)
        elif transform_id == 3:  # rot270 -> rot90
            return torch.rot90(grid_2d, k=1, dims=[-2, -1]).reshape(grid.shape)
        elif transform_id == 4:  # mirror -> mirror
            return torch.flip(grid_2d, dims=[-1]).reshape(grid.shape)
        elif transform_id == 5:  # mirror+rot90 -> mirror+rot270
            return torch.flip(
                torch.rot90(grid_2d, k=3, dims=[-2, -1]), dims=[-1]
            ).reshape(grid.shape)
        elif transform_id == 6:  # mirror+rot180 -> mirror+rot180
            return torch.flip(
                torch.rot90(grid_2d, k=2, dims=[-2, -1]), dims=[-1]
            ).reshape(grid.shape)
        elif transform_id == 7:  # mirror+rot270 -> mirror+rot90
            return torch.flip(
                torch.rot90(grid_2d, k=1, dims=[-2, -1]), dims=[-1]
            ).reshape(grid.shape)

    elif aug_type == "color":
        perm = aug_info
        # invert color permutation
        inv_perm = torch.argsort(perm)
        return inv_perm[grid_2d].reshape(grid.shape)

    return grid
