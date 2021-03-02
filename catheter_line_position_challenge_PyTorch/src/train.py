import pandas as pd
import numpy as np
import os

import random

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import GroupKFold

from albumentations import Compose, RandomResizedCrop, Resize, HorizontalFlip, Normalize
from albumentations.pytorch import ToTensorV2

import cv2
import timm
from tqdm import tqdm

from config import Config

from sklearn.metrics import roc_auc_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            scores.append(score)
        except ValueError:
            pass
    
    avg_score = np.mean(scores)
    return avg_score, scores

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def cv_split(df):
    folds = df.copy()
    Fold = GroupKFold(n_splits=Config.n_folds)
    groups = folds['PatientID'].values
    
    for n, (train_index, val_index) in enumerate(Fold.split(folds,
                                                folds[Config.target_cols],
                                                groups)):
        folds.loc[val_index, 'fold'] = int(n)
    
    folds['fold'] = folds['fold'].astype(int)

    folds.to_csv(os.path.join(Config.data_path, 'folds.csv'), index=False)
    return folds


class TrainDataset(Dataset):

    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_names = df['StudyInstanceUID'].values
        self.labels = df[Config.target_cols].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(Config.train_path, f'{file_name}.jpg')
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = torch.tensor(self.labels[idx]).float()
        return image, label


def transform_image(*, data):
    if data == 'train':
        return Compose([
            #Resize(Config.size, Config.size),
            RandomResizedCrop(Config.size, Config.size, scale=(0.85, 1.0)),
            HorizontalFlip(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(Config.size, Config.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


class Model(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, Config.target_size)

    def forward(self, x):
        x = self.model(x)
        return x


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    scaler = GradScaler()

    model.train()
    progress_bar = tqdm(train_loader)
    losses = []

    for step, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        with autocast():
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        losses.append(loss.item())
        progress_bar.set_description(f'loss: {loss.item(): .5f}')
    
    return np.mean(losses)

def valid_fn(valid_loader, model, criterion, device):
    model.eval()

    progress_bar = tqdm(valid_loader)
    preds = []
    losses = []

    for step, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            y_preds = model(images)
        
        loss = criterion(y_preds, labels) ##
        losses.append(loss.item())

        progress_bar.set_description(f'loss: {loss.item(): .5f}')
        preds.append(y_preds.sigmoid().to('cpu').numpy())

    predictions = np.concatenate(preds)
    return np.mean(losses), predictions


def train_loop(folds, fold):
    train_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] != fold].index

    train_folds = folds.loc[train_idx].reset_index(drop=True)
    val_folds = folds.loc[val_idx].reset_index(drop=True)
    val_labels = val_folds[Config.target_cols].values

    train_dataset = TrainDataset(train_folds, transform=transform_image(data='train'))
    val_dataset = TrainDataset(val_folds, transform=transform_image(data='valid'))

    train_loader = DataLoader(train_dataset, 
                                batch_size=Config.batch_size,
                                shuffle=True,
                                num_workers=Config.num_workers,
                                pin_memory=True,
                                drop_last=True)

    val_loader = DataLoader(val_dataset, 
                                batch_size=Config.batch_size * 2,
                                shuffle=False,
                                num_workers=Config.num_workers,
                                pin_memory=True,
                                drop_last=False)

    model = Model(Config.model_name, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_max, eta_min=Config.min_lr, last_epoch=-1)
    
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0
    best_loss = np.inf
    save_path = os.path.join(Config.output_dir, f'{Config.model_name}_fold{fold}_best.pth')
    for epoch in range(Config.epochs):

        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        avg_val_loss, preds = valid_fn(val_loader, model, criterion, device)

        scheduler.step()

        score, scores = get_score(val_labels, preds)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            
            torch.save({'model':model.state_dict(),
                        'preds':preds},
                        save_path)

    checkpoint = torch.load(save_path)
    for c in [f'pred_{c}' for c in Config.target_cols]:
        val_folds[c] = np.nan
    val_folds[[f'pred_{c}' for c in Config.target_cols]] = checkpoint['preds']

    return val_folds


def main():
    
    train = pd.read_csv(os.path.join(Config.data_path, 'train.csv'))
    
    if Config.debug:
        train = train.sample(n=100, random_state=Config.seed).reset_index(drop=True)

    set_seed(seed=Config.seed)
    folds = cv_split(train)

    def get_result(result_df):
        preds = result_df[[f'pred_{c}' for c in Config.target_cols]].values
        labels = result_df[Config.target_cols].values
        score, scores = get_score(labels, preds)

    if Config.train:
        oof_df = pd.DataFrame()
        for fold in range(Config.n_folds):
            if fold in Config.train_folds:
                _oof_df = train_loop(folds, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                get_result(oof_df)
        get_result(oof_df)
        oof_df.to_csv(os.path.join(Config.output_dir, 'oof_df.csv'), index=False)

if __name__ == '__main__':
    main()