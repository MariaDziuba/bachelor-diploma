import torch
from torch import nn
import importlib
import sys
import os
import sysconfig
import pathlib
import svgutils
import pandas as pd
import ast
import torch
from svgpathtools import svg2paths, wsvg, Path, CubicBezier
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import json
import torch.nn as nn
from random import randrange
import random
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import pydiffvg
import traceback
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
import time
from create_dataset import create_df
from main import Paths_model
from main import create_tenzor
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img = 'feather'
in_dir = './images'
out_dir = '.'

os.system(f'./vtracer-linux --input {in_dir}/{img}.png --output {out_dir}/vtracer_{img}.svg')

df = pd.DataFrame(data={'full_path': [f'{out_dir}/vtracer_{img}.svg'], 'name': [img], 'similarity' :[0]})

df = create_df(df)


model = Paths_model()

checkpoint = torch.load('./checkpoint/paths_wdd_5conv_2fc_best_model_24_04.pt')
model.load_state_dict(checkpoint['model_state_dict'])

if torch.cuda.is_available():
    model.cuda()

model.eval()

temp_img = create_tenzor(df['matrix_padding'][0]).to(device)
outputs = model(temp_img.float())

n_curves = round(outputs.detach().cpu().item())
print(n_curves)

out_yml = ''

with open(r'./template_config.yaml') as file:
    listt = yaml.load(file, Loader=yaml.FullLoader)
    listt['experiment']['path_schedule']['max_path'] = n_curves
    print(listt)
    with open('./modified_config.yaml', 'w') as file:
        yaml.dump(listt, file)
