import torch
import torch.nn.functional as F


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img, sizes=None):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    if sizes is not None:
        coord = make_coord(sizes, flatten=False)
    else:
        coord = make_coord(img.shape[-2:])

    if img is not None:
        rgb = img.view(3, -1).permute(1, 0)
        return coord, rgb

    return coord


def gridy2gridx_flow(gridy, th, tw, offset, tgt_symbol=True):
    gridy = gridy.flip(-1)

    if tgt_symbol:
        gridx = gridy.permute(0,3,1,2)
        gridx = F.interpolate(gridx, size=(th, tw), mode='bilinear', align_corners=False)
        gridx = gridx.permute(0,2,3,1)
    else:
        gridx = gridy + offset
        gridx = gridx.permute(0,3,1,2)
        gridx = F.interpolate(gridx, size=(th, tw), mode='bilinear', align_corners=False)
        gridx = gridx.permute(0,2,3,1)

    return gridx

