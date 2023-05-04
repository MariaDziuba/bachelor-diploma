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

pd.set_option('display.max_columns', None)

def diffvg_svg(path):
    list_path = pydiffvg.svg_to_scene(path)
    return pydiffvg.svg_to_str(*list_path)

def save_info(df, path, name, similarity):
    text = [path, name, similarity]
    temp_data = pd.DataFrame([text], columns=['full_path', 'name', 'similarity'])
    df = pd.concat([df, temp_data], axis=0, ignore_index=True)
    return df

def save_info_parse(df, path, name, matrix, similarity):
    text = [path, name, matrix, similarity]
    temp_data = pd.DataFrame([text], columns=['full_path', 'name', 'matrix', 'similarity'])
    df = pd.concat([df, temp_data], axis=0, ignore_index=True)
    return df

def create_df(df):
    save_dir = "./save/"
    t = 0
    f = 0
    k = 0
    no_conv = 0
    df_diff = pd.DataFrame(columns=['full_path', 'name', 'similarity'])
    for ind in df.index:
        name = df['name'][ind]
        if k % 100 == 0:
            print(k)
        k = k + 1
        try:
            str_svg = diffvg_svg(df['full_path'][ind])
            f = f + 1
        except:
            t = t + 1
            print(traceback.format_exc())
            continue
        fig = svgutils.transform.fromstring(str_svg)
        fig.save(os.path.join(save_dir, name))
        try:
            paths, attributes = svg2paths(os.path.join(save_dir, name))
            wsvg(paths, filename=os.path.join(save_dir, name))
        except:
            no_conv = no_conv + 1
        df_diff = save_info(df_diff, os.path.join(save_dir, name), name, df['similarity'][ind])

    df = df_diff
    err_read = 0
    read = 0
    k = 0
    err_svg = 0
    df_parse = pd.DataFrame(columns=['full_path', 'name', 'matrix', 'similarity'])
    for ind in df.index:
        name = df['name'][ind]

        if k % 100 == 0:
            print(k)
        k = k + 1
        try:
            print(df['full_path'][ind])
            paths, attributes = svg2paths(df['full_path'][ind])
            read = read + 1
        except:
            err_read = err_read + 1
            continue
        #print(df['full_path'][ind])
        list_paths = []
        matrix = []
        paths_new = Path()
        #print(paths)
        for path in paths:
            # list_path = []
            for obj in path:
                str_maxtix = []
                # print(obj)
                try:
                    len(obj)
                except:
                    # print('opop')
                    # print(obj)
                    continue
                if len(obj) == 2:
                    # paths_new.append(obj)
                    x = 0
                    for point in obj:
                        num_point_x = float(point.real)
                        num_point_x = round(num_point_x, 2)
                        str_maxtix.append(num_point_x)
                        num_point_y = float(point.imag)
                        num_point_y = round(num_point_y, 2)
                        str_maxtix.append(num_point_y)

                        str_maxtix.append(num_point_x)
                        str_maxtix.append(num_point_y)
                        if x == 0:
                            point1 = complex(num_point_x, num_point_y)
                            point2 = complex(num_point_x, num_point_y)
                            x += 1
                        elif x == 1:
                            point3 = complex(num_point_x, num_point_y)
                            point4 = complex(num_point_x, num_point_y)
                    paths_new.append(CubicBezier(point1, point2, point3, point4))
                elif len(obj) == 3:
                    # paths_new.append(obj)
                    x = 0
                    for point in obj:
                        num_point_x = float(point.real)
                        num_point_x = round(num_point_x, 2)
                        str_maxtix.append(num_point_x)
                        num_point_y = float(point.imag)
                        num_point_y = round(num_point_y, 2)
                        str_maxtix.append(num_point_y)
                        if x == 0:
                            x += 1
                            str_maxtix.append(num_point_x)
                            str_maxtix.append(num_point_y)
                            point1 = complex(num_point_x, num_point_y)
                            point2 = complex(num_point_x, num_point_y)
                        elif x == 1:
                            x += 1
                            point3 = complex(num_point_x, num_point_y)
                        elif x == 2:
                            point4 = complex(num_point_x, num_point_y)
                    paths_new.append(CubicBezier(point1, point2, point3, point4))

                elif len(obj) == 4:
                    paths_new.append(obj)
                    for point in obj:
                        num_point = float(point.real)
                        num_point = round(num_point, 2)
                        str_maxtix.append(num_point)
                        num_point = float(point.imag)
                        num_point = round(num_point, 2)
                        str_maxtix.append(num_point)

                else:
                    print(df['full_path'][ind])
                    print(obj)
                if str_maxtix:
                    # print(len(str_maxtix))
                    matrix.append(str_maxtix)

        # print(matrix)
        if not matrix:
            print('oi:', df['full_path'][ind])
            err_svg = err_svg + 1
            # print(matrix)
            # list_paths.append(list_path)
        # paths_new = Path(list_paths)
        try:
            wsvg(paths_new, filename=os.path.join(save_dir, name))
            p = 0
        except:
            err_svg = err_svg + 1
            continue
        df_parse = save_info_parse(df_parse, os.path.join(save_dir, name), name, matrix, df['similarity'][ind])
    #print(read, err_read, err_svg)

    size = 20000
    df = df_parse
    #print(df)
    import json
    df['num_obj'] = None
    for ind in df.index:
        mat = json.loads(str(df['matrix'][ind]))
        df.at[ind, 'num_obj'] = len(mat)

    df_drop = df.copy()
    df_drop = df_drop.drop(df_drop[df_drop['num_obj'] > size].index)
    df_drop = df_drop.drop(df_drop[df_drop['num_obj'] < 10].index)
    # заполнение 0
    padding = [0, 0, 0, 0, 0, 0, 0, 0]
    df_drop['matrix_padding'] = None
    for ind in df_drop.index:
        matrix = json.loads(str(df_drop['matrix'][ind]))
        while len(matrix) < size:
            matrix.append(padding)
        df_drop.at[ind, 'matrix_padding'] = matrix
    df = df_drop
    return df

def main():
    in_dir = './final_logomark_png_10_contours_svg'
    listdir = os.listdir(in_dir)

    listdir = list(set(map(lambda x: x.replace('._', ''), listdir)))
    #filenames=list(map(lambda x: Path(x).stem, listdir))
    df = pd.DataFrame(data={'full_path': list(map(lambda x: f'{in_dir}/{x}', listdir)) , 'name': listdir, 'similarity' :[_ for _ in range(len(listdir))]})
    #df = pd.DataFrame(data={'full_path': [path], 'name': [name], 'similarity' :[0]})
    #print(df)

    df = create_df(df)

    labels = pd.read_csv('./final_logomark_png_10/labels.csv')
    print(labels)
    df['img'] = df['name'].apply(lambda x: pathlib.Path(x).stem)
    print(df.info())
    print(labels.info())
    labels['paths'] = labels['paths'].astype(str)
    final = pd.merge(df, labels, on='img', how='inner')
    final['paths'] = final['paths'].astype(int)
    print(final)
    final.to_csv(os.path.join(in_dir, 'labels.csv'), index=False)

if __name__ == "__main__":
   main()
