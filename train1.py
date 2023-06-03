from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import json
import torch.nn as nn
from random import randrange
import random
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from train2 import create_tenzor, Mega_model

df = pd.read_csv(r'/Users/maria/data.csv')
data = df[['name', 'matrix_padding', 'similarity']]

def add_data(data, list_data):
    temp_data = pd.DataFrame([list_data], columns = ['name_1', 'name_2', 'matrix_1', 'matrix_2', 'sim'])
    data = pd.concat([data, temp_data], axis = 0, ignore_index=True)
    
    return data

sim = -1
new_data = pd.DataFrame(columns = ['name_1', 'name_2', 'matrix_1', 'matrix_2', 'sim'])
for ind in data.index:
    if sim != data['similarity'][ind]:
        sim = data['similarity'][ind]
    else:
        continue
    if sum(x == sim for x in list(data['similarity'])) > 1:
        sim_image = 1
        for ind_j in data[data['similarity'] == sim].index:
            if ind != ind_j:
                list_data = [data['name'][ind], data['name'][ind_j], data['matrix_padding'][ind], data['matrix_padding'][ind_j], sim_image]
                new_data = add_data(new_data, list_data)
    else:
        sim_image = 0
        while True:
            ind_j = random.randint(0, len(data) - 1)
            if ind != ind_j:
                list_data = [data['name'][ind], data['name'][ind_j], data['matrix_padding'][ind], data['matrix_padding'][ind_j], sim_image]
                new_data = add_data(new_data, list_data)
            else:
                continue
            if random.random() > 0:
                break

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ContrastiveLoss_v1(torch.nn.Module):
    def __init__(self, m = 2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.m = m

    def forward(self, y1, y2, d = 0):
    
        euc_dist = torch.nn.functional.pairwise_distance(y1, y2)

        if d == 0:
            return torch.mean(torch.pow(euc_dist, 2))
        else:  # d == 1
            delta = self.m - euc_dist
            delta = torch.clamp(delta, min=0.0, max=None)
            return torch.mean(torch.pow(delta, 2))

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=1.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss = torch.mean(1/2*(label) * torch.pow(dist, 2) +
                                      1/2*(1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss
    
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
def dist(input_1, input_2):
    return 1 - cos(input_1, input_2)

        
model = Mega_model().to(device)
#margin = 500
margin = 0.1
epox_num = 8
epox_list = [i for i in range(epox_num)]
criterion = ContrastiveLoss(margin)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

train, test = train_test_split(new_data, test_size = 0.2 , random_state = 322 )
dataset = {'train':train, 'test':test}
len(train), len(test)

tqdm.pandas()

def train_model(model, criterion, optimizer, margin, num_epochs=3):
    loss_list = {'train' : [], 'test':[]}
    acc_list = {'train' : [], 'test':[]}
    rec_list = {'train' : [], 'test':[]}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in dataset:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_corrs_tp = 0
            running_corrs_fn = 0
            for ind, inputs in tqdm(dataset[phase].iterrows()):
                name_1, name_2, input_1, input_2, labels = inputs['name_1'], inputs['name_2'], inputs['matrix_1'], inputs['matrix_2'], inputs['sim']
                input_1 = create_tenzor(input_1).to(device)
                input_2 = create_tenzor(input_2).to(device)
                labels = torch.tensor(labels).to(device)
                outputs_1 = model(input_1)
                outputs_2 = model(input_2)
                #print(outputs_1)
                #print(outputs_2)
                distantion = dist(outputs_1, outputs_2)
                
                loss = criterion(distantion, labels)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
                    #model.float()
                    optimizer.step()
                running_loss += loss.item()# * inputs.size(0)
                if distantion>margin:
                    preds = 0
                else:
                    preds = 1
                #print(distantion, preds, labels)
                if preds == labels and labels == 1:
                    running_corrs_tp += 1
                elif preds != labels and labels == 1:
                    running_corrs_fn += 1
                
                if preds == labels:
                    running_corrects += 1
            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_corrects / len(dataset[phase])
            epoch_rec_1 = running_corrs_tp / (running_corrs_tp + running_corrs_fn)
            print('{} loss: {:.4f}, acc: {:.4f}, rec_1: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc, 
                                                        epoch_rec_1
                                                        ))
            loss_list[phase].append(epoch_loss)
            acc_list[phase].append(epoch_acc)
            rec_list[phase].append(epoch_rec_1)
    return loss_list, acc_list, rec_list


loss, acc, rec_1 = train_model(model, criterion, optimizer, margin, epox_num)

train_model(model, criterion, optimizer, margin, 8)

def serch_sim(df, model, name_image):
    model.eval()
    df_sim = df.copy()
    df_sim['sim'] = None
    curr_img = df_sim['matrix_padding'][df_sim['name'] == name_image]
    try:
        curr_img = list(curr_img)[0]
    except:
        print(curr_img)
        return None
    curr_img = create_tenzor(curr_img).to(device)
    curr_embed = model(curr_img)
    df_sim = df_sim[df.name != name_image]
    
    for ind in df_sim.index:
        temp_img = create_tenzor(df_sim['matrix_padding'][ind]).to(device)
        temp_embed = model(temp_img)
        similarity = dist(curr_embed, temp_embed)
        df_sim.at[ind, 'sim'] = similarity.item()
    return df_sim

