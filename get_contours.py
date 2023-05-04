import torch
import pydiffvg
import copy
import os
from pathlib import Path

def get_contours(shape_groups):
      shape_groups_new = copy.deepcopy(shape_groups)
      with torch.no_grad():
        for shape_group in shape_groups_new:
            if isinstance(shape_group.fill_color, torch.Tensor):
                shape_group.fill_color[0] = 1.
                shape_group.fill_color[1] = 1.
                shape_group.fill_color[2] = 1.
                strike_color = torch.Tensor([0.,0.,0.,1.])
                shape_group.stroke_color = strike_color
            elif isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
                shape_group.fill_color.begin = 1.
                shape_group.fill_color.end = 1.
                shape_group.fill_color.stop_colors = 1.
                begin = torch.Tensor(0.)
                end = torch.Tensor(0.)
                stop_colors = torch.Tensor(0.)
                shape_group.stroke_color.begin = begin
                shape_group.stroke_color.end = end
                shape_group.stroke_color.stop_colors = stop_colors
      
      return shape_groups_new

def main():
    in_dir = './final_logomark_png_10_vtracer'
    listdir = os.listdir(in_dir)

    listdir = set(map(lambda x: x.replace('._', ''), listdir))
    filenames=list(map(lambda x: Path(x).stem, listdir))

    print(filenames)

    out_dir = './final_logomark_png_10_contours_svg'

    for f in filenames:
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(f'{in_dir}/{f}.svg')
        shape_groups_new = get_contours(shape_groups)
        pydiffvg.save_svg(f'{out_dir}/{f}.svg',canvas_width, canvas_height,shapes, shape_groups_new)

if __name__ == "__main__":
   main()