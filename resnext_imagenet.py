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


class VectorizedImageNetDataset(Dataset):
    """Vectorized ImageNet dataset"""

    def __init__(self, csv_df, root_dir, transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = csv_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        labels = self.landmarks_frame
        img_path = os.path.join(self.root_dir,
                                labels.iloc[int(idx)]['path'])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        landmarks = labels.iloc[int(idx)]['label']
        landmarks = torch.from_numpy(np.array([landmarks]))

        if self.transform:
            image = self.transform(image=image)["image"]
        return image, landmarks


transform = A.Compose([
    A.Resize(224, 224),
    A.RandomRotate90(),
    A.Transpose(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    ToTensorV2(),
])

batch_size = 16
random_seed = 42


def insert_index(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.reset_index(drop=True, inplace=True)
    df['idx'] = range(0, len(df))
    df.set_index('idx', inplace=True)
    return df


train_csv = insert_index('/Volumes/BANQ/THESIS/imagenet_jpg/labels.csv')

validation_split = .05
shuffle_dataset = True
random_seed = 42

dataset = VectorizedImageNetDataset(csv_df=train_csv, root_dir='/Volumes/BANQ/THESIS/imagenet_jpg', transform=transform)

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

for name, param in model.named_parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 10),
    nn.Sigmoid()
)

params_to_optimize = [param for _, param in model.named_parameters() if param.requires_grad]
optimizer = optim.SGD(params_to_optimize, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


def training(epochs: int, data_loader):
    for epoch in range(epochs):
        running_loss = 0.0
        number_of_sub_epochs = 0
        start = time.time()
        y_true = []
        y_pred = []
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            # inputs, labels = data[0].to(device), data[1]
            # print(inputs.size())
            y_true.extend(labels.numpy())
            optimizer.zero_grad()
            # inputs = torch.from_numpy(np.einsum('ijkl->iljk', inputs))
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            loss = criterion(outputs, labels.reshape(-1))
            loss.backward()
            optimizer.step()
            number_of_sub_epochs += 1
            running_loss += loss.item()
        end = time.time()
        print("Epoch {}: Loss: {} Time: {} ".format(epoch, running_loss / number_of_sub_epochs, (end - start)))
        print('accuracy: ', accuracy_score(y_true, y_pred))
        print('f1_score: ', f1_score(y_true, y_pred, average='weighted'))
        print('precision: ', precision_score(y_true, y_pred, average='weighted'))
        print('recall: ', recall_score(y_true, y_pred, average='weighted'))
    print('Finished Training')

training(epochs=5, data_loader=train_loader)
