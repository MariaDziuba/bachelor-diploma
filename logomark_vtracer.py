import os
import zipfile
from pathlib import Path
import ntpath
from PIL import Image

in_dir = 'final_logomark_png_10'
listdir = os.listdir(in_dir)

listdir = set(map(lambda x: x.replace('._', ''), listdir))
filenames=list(map(lambda x: Path(x).stem, listdir))

print(filenames)

out_dir = 'final_logomark_png_10_vtracer'

for f in filenames:
    os.system(f'vtracer --input {in_dir}/{f}.jpg --output {out_dir}/{f}.svg')