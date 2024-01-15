"""
Used to generate tag groups that allow for unique pose estimation.
"""

from moms_apriltag import TagGenerator2
from matplotlib import pyplot as plt
import numpy as np
import cv2
from pathlib import Path

tg = TagGenerator2("tag36h11")
tag = tg.generate(4, scale=3)

def create_group(group_idx, center_scale = 3, outer_scale = 2):
    assert outer_scale <= center_scale
    center_tag = tg.generate(2*group_idx, scale=center_scale)
    upper_tag = tg.generate(2*group_idx+1, scale=outer_scale)
    width = center_tag.shape[1] + 2*center_scale
    height = center_tag.shape[0] + upper_tag.shape[0] + 2*center_scale + outer_scale
    group = np.zeros((height, width), dtype=np.uint8)
    group.fill(255)
    group[outer_scale:upper_tag.shape[0]+outer_scale, -upper_tag.shape[1]-center_scale:-center_scale] = upper_tag
    center_y_start = upper_tag.shape[0] + outer_scale + center_scale
    group[center_y_start:center_y_start+center_tag.shape[0], center_scale:center_scale+center_tag.shape[1]] = center_tag
    
    return group

def create_printable_group(group_idx, center_scale = 3, outer_scale = 2, dpi=300, tag_size_mm=165):
    group = create_group(group_idx, center_scale, outer_scale)

    tag_size_in = tag_size_mm / 25.4
    tag_width_px = int(tag_size_in * dpi)
    # This tag width corresponds to only the tag itself, not the white border
    full_to_tag_ratio = group.shape[1] / (group.shape[1] - 2*center_scale)
    image_width_px = int(tag_width_px * full_to_tag_ratio)
    print(f"Tag Width: {tag_width_px} px ({tag_size_mm} mm). Image Width: {image_width_px} px ({image_width_px / dpi} in / {image_width_px / dpi * 25.4} mm)")
    tag_2_width_px = int(tag_width_px * outer_scale / center_scale)
    tag_2_size_mm = tag_size_mm * outer_scale / center_scale
    print(f"Tag 2 Width: {tag_2_width_px} px ({tag_2_size_mm} mm)")
    upscale_ratio = image_width_px / group.shape[1]
    # Upscale the image to the desired width
    group = cv2.resize(group, (0,0), fx=upscale_ratio, fy=upscale_ratio, interpolation=cv2.INTER_NEAREST)
    print(f"Group Height: {group.shape[0]} px ({group.shape[0] / dpi} in / {group.shape[0] / dpi * 25.4} mm)")
    
    # Place the group idx in the top left corner
    text = f"{group_idx}"
    text_size_pixels = int((72 / 72) * dpi)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = text_size_pixels / 30  # 30 is a base size for OpenCV fonts
    font_thickness = int(6/72 * dpi) # 12 is a base thickness for OpenCV fonts
    offset = int(12/72 * dpi)
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    cv2.putText(group, text, (offset, offset+text_height), font, font_scale, (0, 0, 0), font_thickness)

    return group

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, required=True, help='Directory to save the tag groups.')
    parser.add_argument('--num-groups', type=int, default=24, help='Number of tag groups to generate.')
    parser.add_argument('--center-scale', type=int, default=3, help='Relative side width of center tag.')
    parser.add_argument('--outer-scale', type=int, default=2, help='Relative side width of top tag.')
    parser.add_argument('--dpi', type=int, default=72, help='DPI of the output images.')
    parser.add_argument('--tag-size-mm', type=int, default=130, help='Size of the center tag in mm.')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    num_groups = args.num_groups
    center_scale = args.center_scale
    outer_scale = args.outer_scale
    dpi = args.dpi
    tag_size_mm = args.tag_size_mm

    for i in range(num_groups):
        group = create_printable_group(i, center_scale=center_scale, outer_scale=outer_scale, dpi=dpi, tag_size_mm=tag_size_mm)
        # Save the group to disk
        out_file = out_dir / f"group_{i}.png"
        cv2.imwrite(out_file.absolute().as_posix(), group)