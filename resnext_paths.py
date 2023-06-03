import os
import zipfile
import pandas as pd
from skimage import io, transform
import albumentations as A
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
import copy
import cv2
from torchvision.models import *
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SvgPathsDataset(Dataset):
    """Svg Number of Paths dataset."""
    
    def __init__(self, csv_file, root_dir, transform):
        self.landmarks_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        labels = self.landmarks_frame
        img_name = labels.iloc[idx]['img']
        # path to image
        img_path = os.path.join(self.root_dir,
                                img_name.strip() + '.jpg')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.einsum('kli->ikl', image)
        # pillow_image = Image.open(img_path)
        # image = np.array(pillow_image)

        landmarks = labels.loc[labels['img'] == img_name, 'paths'].iloc[0]
        landmarks = np.array([landmarks])

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, landmarks


transform = A.Compose([
    A.RandomRotate90(),
    A.Transpose(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
])

batch_size = 16
validation_split = .05
shuffle_dataset = True
random_seed = 12

c_f = pd.read_csv('/Volumes/KINGSTON/THESIS/final_logomark_png_10/labels.csv')
dataset = SvgPathsDataset(csv_file=c_f, root_dir='/Volumes/KINGSTON/THESIS/final_logomark_png_10', transform=transform)

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

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 10),
    nn.Sigmoid()
)

checkpoint = torch.load('/Volumes/KINGSTON/THESIS/models/imagenet_best_model_15_01.pt',
                        map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# criterion = checkpoint['criterion']
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# model.fc = nn.Sequential(
#     nn.Linear(2048, 512),
#     nn.LeakyReLU(),
#     nn.Linear(512, 256),
#     nn.LeakyReLU(),
#     nn.Linear(256, 1),
#     nn.LeakyReLU()
# )
#

for name, param in model.named_parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 1),
    nn.ReLU()
)

for name, param in model.named_parameters():
  if param.requires_grad:
    print(name)


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

params_to_optimize = [param for _, param in model.named_parameters() if param.requires_grad]
optimizer = optim.Adam(params_to_optimize, lr=0.0001)
# optimizer = optim.SGD(params_to_optimize, lr=0.0001, momentum=0.9)
# criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
criterion = RMSLELoss()
# model.fc = nn.Sequential(
#     nn.Linear(2048, 1),
#     nn.ReLU()
# )

train_loss = []
val_loss = []
train_smape = []
val_smape = []


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
    plt.ylabel('Smape')
    plt.legend()
    plt.savefig(f'/Volumes/KINGSTON/THESIS/models/paths_1fc_18_01_smape_epoch_{epoch}.png')
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        val_loss, color='red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'/Volumes/KINGSTON/THESIS/models/paths_1fc_18_01_loss_epoch_{epoch}.png')


from torchmetrics import SymmetricMeanAbsolutePercentageError

smape = SymmetricMeanAbsolutePercentageError()

from torchmetrics import MeanAbsolutePercentageError

mean_abs_percentage_error = MeanAbsolutePercentageError()

from torchmetrics import MeanSquaredError

mean_squared_error = MeanSquaredError()


def validation(data_loader, min_val_loss, epoch):
    if torch.cuda.is_available():
        model.cuda() 
    cur_val_loss = 2000
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
            inputs = torch.from_numpy(np.einsum('ijkl->iljk', inputs))
            y_true.extend(labels.numpy())
            inputs, labels = inputs, labels
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            predicted = outputs
            y_pred.extend(predicted.cpu().numpy())
            loss = criterion(outputs, labels.float())
            number_of_sub_epochs += 1
            running_loss += loss.item()
    end = time.time()
    cur_val_loss = running_loss / number_of_sub_epochs
    val_loss.append(cur_val_loss)
    if cur_val_loss < min_val_loss:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion': criterion,
            'loss': cur_val_loss
        }, '/Volumes/KINGSTON/THESIS/models/paths_1fc_best_model_18_01.pt')
        # save model to disk
        min_val_loss = cur_val_loss
    if (epoch % 10 == 0) and (not epoch == 0):
        save_plots(epoch)
    print("Validation Loss: {} Time: {} ".format(cur_val_loss, (end - start)))
    smape_ = smape(torch.FloatTensor(y_pred), torch.FloatTensor(y_true))
    val_smape.append(smape_)
    print('smape: ', smape_)
    print('mape: ', mean_abs_percentage_error(torch.FloatTensor(y_pred), torch.FloatTensor(y_true)))
    print('mse: ', mean_squared_error(torch.FloatTensor(y_pred), torch.FloatTensor(y_true)))
    return min_val_loss


eps = 0.00001


def training(epochs: int, data_loader, validation_loader):
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    loss_last = 20000
    loss_cur = 20001
    min_val_loss = 2000
    for epoch in range(epochs):
        if np.abs(loss_last - loss_cur) < eps:
            break
        running_loss = 0.0
        number_of_sub_epochs = 0
        start = time.time()
        y_true = []
        y_pred = []
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = torch.from_numpy(np.einsum('ijkl->iljk', inputs))
            y_true.extend(labels.numpy())
            inputs, labels = inputs, labels
            print(torch.any(torch.isnan(inputs)))
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            # print(outputs)
            predicted = outputs
            y_pred.extend(predicted.detach().numpy())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            number_of_sub_epochs += 1
            running_loss += loss.item()
        end = time.time()
        loss_last = loss_cur
        loss_cur = running_loss / number_of_sub_epochs
        train_loss.append(loss_cur)
        print(f'EPOCH {epoch}')
        print(np.concatenate(y_pred).ravel().tolist())
        y_pred = np.concatenate(y_pred).ravel().tolist()
        print(np.concatenate(y_true).ravel().tolist())
        y_true = np.concatenate(y_true).ravel().tolist()
        print("Training Loss: {} Time: {} ".format(running_loss / number_of_sub_epochs, (end - start)))
        smape_ = smape(torch.FloatTensor(y_pred), torch.FloatTensor(y_true))
        train_smape.append(smape_)
        print('smape: ', smape_)
        print('mape: ', mean_abs_percentage_error(torch.FloatTensor(y_pred), torch.FloatTensor(y_true)))
        print('mse: ', mean_squared_error(torch.FloatTensor(y_pred), torch.FloatTensor(y_true)))
        min_val_loss = validation(validation_loader, min_val_loss, epoch)
    print('Finished Training')


print(model.eval())
model.train()

training(epochs=101, data_loader=train_loader, validation_loader=validation_loader)
