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
pd.set_option('display.max_columns', None)
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Mega_model(nn.Module):
    def __init__(self, num_emb=5000):
        super(Mega_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), stride=(1, 2))
        )
        self.act_l1 = nn.ReLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(800, 2), stride=(1, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(400, 2), stride=(1, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(200, 2), stride=(1, 1))
        )
        self.act_l2 = nn.ReLU()
        self.pooling = nn.AvgPool2d(kernel_size=(3, 1))

        # torch.flatten
        self.fc1 = nn.Linear(in_features=6201, out_features=6000)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=6000, out_features=5500)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=5500, out_features=num_emb)

    def forward(self, input_matrix):
        x = self.conv1(input_matrix)
        x = self.act_l1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.act_l2(x)
        x = self.pooling(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

class Paths_model(nn.Module):
    def __init__(self):
        super().__init__()
        m = Mega_model()
        m.load_state_dict(torch.load('model_weights.pth'))
        self.emb_extractor = m
        self.bn1 = nn.BatchNorm1d(num_features=1)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=1)
        self.act1 = nn.LeakyReLU(0.3)
        
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=1)
        self.act2 = nn.LeakyReLU(0.3)

        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=1)
        self.act3 = nn.LeakyReLU(0.3)

        self.conv4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(num_features=1)
        self.act4 = nn.LeakyReLU(0.3)

        self.conv5 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(num_features=1)
        self.act5 = nn.LeakyReLU(0.3)

        self.avgpool = nn.AdaptiveAvgPool1d(625)

        self.fc1 = nn.Linear(in_features=625, out_features=250)
        self.act6 = nn.LeakyReLU(0.3)

        self.fc2 = nn.Linear(in_features=250, out_features=1)
        self.act7 = nn.LeakyReLU(0.3)

    def forward(self, x):
        #print(x.shape)
        x = self.emb_extractor(x)
        #print('embed', torch.isnan(x).any())
        x.unsqueeze_(0)
        x.unsqueeze_(0)
        #print(x.shape)
        x = self.bn1(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act1(x)
        #print(x.shape)
        #print('conv1', torch.isnan(x).any())
        x = self.conv2(x)
        x = self.bn3(x)
        #x += shortcut
        x = self.act2(x)
#        x = self.pool2(x)
        #print(x.shape)

        #print('conv2', torch.isnan(x).any())
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.act3(x)
        #print(x.shape)
        #print('conv3', torch.isnan(x).any())
        x = self.conv4(x)
        x = self.bn5(x)
        x = self.act4(x)
        #print(x.shape)

        #print('conv4', torch.isnan(x).any())
        x = self.conv5(x)
        x = self.bn6(x)
        x = self.act5(x)
        #print(x.shape)
        #print('conv5', torch.isnan(x).any())

        x = self.avgpool(x)
        #print(x.shape)

        x = self.fc1(x)
        x = self.act5(x)
        #print(x.shape)
        #print('fc1', torch.isnan(x).any())
        x = self.fc2(x)
        x = self.act7(x)
        #print(x.shape)
        #print('fc2', torch.isnan(x).any())
        return x

class SvgMatrixDataset(Dataset):

    def __init__(self, csv_file):
        self.df = csv_file

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        temp_img = create_tenzor(self.df['matrix_padding'][idx]).to(device)
        labels = self.df
        img_name = labels.iloc[idx]['img']
        landmarks = labels.loc[labels['img'] == img_name, 'paths'].iloc[0]
        landmarks = np.array([landmarks])
        return temp_img, landmarks
    
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

class MSE_Squared_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(pred, actual)**2
    
def create_tenzor(input_matrix):
    input_matrix = json.loads(str(input_matrix))
    input_matrix = torch.tensor([input_matrix])
    input_matrix = input_matrix.to(torch.float)
    return input_matrix


def main():
    model = Paths_model()

    for param in model.emb_extractor.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    if torch.cuda.is_available():
        model.cuda()

    print(model.eval())

    batch_size = 1
    validation_split = .05
    shuffle_dataset = True
    random_seed = 12

    df = pd.read_csv('./final_logomark_png_10_contours_svg/labels.csv')
    dataset = SvgMatrixDataset(csv_file=df)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    train_rmsle_loss = []
    val_rmsle_loss = []
    train_smape = []
    val_smape = []
    train_mse = []
    val_mse = []
    # uncomment this if you want to resume training from checkpoint
    #checkpoint = torch.load('./checkpoint/paths_1fc_best_model_18_01.pt')
    #params_to_optimize = [param for _, param in model.named_parameters() if param.requires_grad]
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer = optim.Adam(params_to_optimize, lr=0.000001)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #criterion = checkpoint['criterion']
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

    # uncomment this if you want to start training from scratch
    params_to_optimize = [param for _, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(params_to_optimize, lr=0.001)
    criterion = RMSLELoss()
    criterion = MSE_Squared_loss()

    def save_plots(epoch):
        # smape plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_smape, color='blue', linestyle='-',
            label='train smape'
        )
        plt.plot(
            val_smape, color='green', linestyle='-',
            label='validation smape'
        )
        plt.xlabel('Epochs')
        plt.ylabel('SMAPE')
        plt.legend()
        plt.savefig(f'./plots/paths_wdd_5conv_2fc_24_04_smape_epoch_{epoch}.png')
        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_rmsle_loss, color='orange', linestyle='-',
            label='train rmsle loss'
        )
        plt.plot(
            val_rmsle_loss, color='red', linestyle='-',
            label='validation rmsle loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('RMSLE Loss')
        plt.legend()
        plt.savefig(f'./plots/paths_wdd_5conv_2fc_24_04_rmsle_loss_epoch_{epoch}.png')
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_mse, color='blueviolet', linestyle='-',
            label='train mse'
        )
        plt.plot(
            val_mse, color='hotpink', linestyle='-',
            label='validation mse'
        )
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(f'./plots/paths_wdd_5conv_2fc_24_04_mse_epoch_{epoch}.png')

    from torchmetrics import SymmetricMeanAbsolutePercentageError

    smape = SymmetricMeanAbsolutePercentageError()

    from torchmetrics import MeanAbsolutePercentageError

    mean_abs_percentage_error = MeanAbsolutePercentageError()

    from torchmetrics import MeanSquaredError

    mean_squared_error = MeanSquaredError()

    def validation(data_loader, min_val_loss, epoch):
        if torch.cuda.is_available():
            model.cuda()
        cur_val_loss = 200000
        model.eval()
        running_loss = 0.0
        number_of_sub_epochs = 0
        start = time.time()
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                y_true.extend(labels.numpy())
                inputs, labels = inputs, labels
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())
                predicted = outputs
                predicted = predicted.view(-1, 1)
                #print('inputs', inputs.cpu().numpy())
                #print('labels', labels.cpu().numpy())
                #print('predicted', predicted.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                loss = criterion(outputs, labels.float())
                number_of_sub_epochs += 1
                running_loss += loss.item()
        end = time.time()
        cur_val_loss = running_loss / number_of_sub_epochs
        val_rmsle_loss.append(cur_val_loss)
        if cur_val_loss < min_val_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion,
                'loss': cur_val_loss
            }, './checkpoint/paths_wd_5conv_2fc_best_model_24_04.pt')
            # save model to disk
            min_val_loss = cur_val_loss
        if (epoch % 10 == 0) and (not epoch == 0):
            save_plots(epoch)
        #print('y_pred', y_pred.shape)
        #print('y_true', y_true.shape)
        print(np.concatenate(y_pred).ravel().tolist())
        y_pred = np.concatenate(y_pred).ravel().tolist()
        print(np.concatenate(y_true).ravel().tolist())
        y_true = np.concatenate(y_true).ravel().tolist()
        print("Validation Loss: {} Time: {} ".format(cur_val_loss, (end - start)))
        smape_ = smape(torch.FloatTensor(y_pred), torch.FloatTensor(y_true))
        val_smape.append(smape_)
        mse_ = mean_squared_error(torch.FloatTensor(y_pred), torch.FloatTensor(y_true))
        val_mse.append(mse_)
        print('smape: ', smape_)
        print('mape: ', mean_abs_percentage_error(torch.FloatTensor(y_pred), torch.FloatTensor(y_true)))
        print('mse: ', mse_)
        return min_val_loss

    eps = 0.00001

    def training(epochs: int, data_loader, validation_loader):
        if torch.cuda.is_available():
            model.cuda()
        model.train()
        loss_last = 2000000
        loss_cur = 2000100
        min_val_loss = 200000
        for epoch in range(epochs):
            if np.abs(loss_last - loss_cur) < eps:
                break
            running_loss = 0.0
            number_of_sub_epochs = 0
            start = time.time()
            y_true = []
            y_pred = []
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                y_true.extend(labels.numpy())
                inputs, labels = inputs, labels
                #print(torch.any(torch.isnan(inputs)))
                inputs, labels = inputs.to(device), labels.to(device)
                # print(inputs.size())
                optimizer.zero_grad()
                outputs = model(inputs.float())
                # print(outputs)
                predicted = outputs
                predicted = predicted.view(-1, 1)
                y_pred.extend(predicted.detach().cpu().numpy())
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                number_of_sub_epochs += 1
                running_loss += loss.item()
            end = time.time()
            loss_last = loss_cur
            loss_cur = running_loss / number_of_sub_epochs
            train_rmsle_loss.append(loss_cur)
            print(f'EPOCH {epoch}')
            #print(np.concatenate(y_pred).ravel().tolist())
            y_pred = np.concatenate(y_pred).ravel().tolist()
            #print(np.concatenate(y_true).ravel().tolist())
            y_true = np.concatenate(y_true).ravel().tolist()
            #print("Training Loss: {} Time: {} ".format(running_loss / number_of_sub_epochs, (end - start)))
            smape_ = smape(torch.FloatTensor(y_pred), torch.FloatTensor(y_true))
            train_smape.append(smape_)
            print('smape: ', smape_)
            print('mape: ', mean_abs_percentage_error(torch.FloatTensor(y_pred), torch.FloatTensor(y_true)))
            print('mse: ', mean_squared_error(torch.FloatTensor(y_pred), torch.FloatTensor(y_true)))
            min_val_loss = validation(validation_loader, min_val_loss, epoch)
        print('Finished Training')

    training(epochs=201, data_loader=train_loader, validation_loader=validation_loader)

if __name__ == "__main__":
   main()
