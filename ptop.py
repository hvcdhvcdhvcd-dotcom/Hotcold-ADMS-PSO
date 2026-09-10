"""Render a four-patch ADMS-PSO particle on every target person."""
import torch
import torch.nn.functional as F


class ParticleToPatch:
    def __init__(self, patch_size, cell_size=16):
        self.ratio_h, self.cell_size, self.temperature = patch_size, cell_size, 0.3

    def __call__(self, particle, targets, imgs):
        shapes, positions = particle
        device, dtype = imgs.device, imgs.dtype
        _, channels, height, width = imgs.shape
        patches, masks = torch.zeros_like(imgs), torch.zeros_like(imgs)
        for target in targets:
            image_index = int(target[0].item())
            box_x, box_y = target[2] * width, target[3] * height
            box_w, box_h = target[4] * width, target[5] * height
            side = max(1, int(float(box_h * self.ratio_h)))
            left, top = box_x - box_w / 2, box_y - box_h / 2
            for shape, position in zip(shapes.to(device), positions.to(device)):
                # Nearest-neighbour upsampling keeps the 3x3 binary pattern.
                mask = F.interpolate(shape.to(dtype)[None, None], size=(side, side), mode='nearest')[0].expand(channels, -1, -1)
                center_x, center_y = int(float(left + position[0] * box_w)), int(float(top + position[1] * box_h))
                x0, y0, x1, y1 = center_x - side // 2, center_y - side // 2, center_x - side // 2 + side, center_y - side // 2 + side
                cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
                if cx0 >= cx1 or cy0 >= cy1:
                    continue
                sx0, sy0 = cx0 - x0, cy0 - y0
                sx1, sy1 = sx0 + cx1 - cx0, sy0 + cy1 - cy0
                source_mask = mask[:, sy0:sy1, sx0:sx1]
                destination_mask = masks[image_index, :, cy0:cy1, cx0:cx1]
                patches[image_index, :, cy0:cy1, cx0:cx1] = torch.where(source_mask.bool(), torch.full_like(source_mask, self.temperature), patches[image_index, :, cy0:cy1, cx0:cx1])
                masks[image_index, :, cy0:cy1, cx0:cx1] = torch.maximum(destination_mask, source_mask)
        return patches, masks
