import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import transformers
import numpy as np
import os

import pandas as pd
import config as cfg
from transformers import AdamW
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def random_seed(seed=cfg.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BERTDataSet(Dataset):

    def __init__(self, sentences, targets, tokenizer):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        bert_sens = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=cfg.MAX_LEN,
            pad_to_max_length=True,
            return_attention_mask=True)

        ids = torch.tensor(bert_sens['input_ids'], dtype=torch.long)
        mask = torch.tensor(bert_sens['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(bert_sens['token_type_ids'], dtype=torch.long)

        target = torch.tensor(self.targets[idx], dtype=torch.float)

        return {
            'ids': ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets': target
        }


def loss_fn(output, target):
    return torch.sqrt(nn.MSELoss()(output, target))


def train_fn(train_loader, model, optimizer):
    model.train()

    predictions = []
    targets = []

    for batch in train_loader:
        losses = []

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            ids = batch["ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            output = model(ids, mask)
            output = output["logits"].squeeze(-1)

            target = batch["targets"].to(device, non_blocking=True)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            predictions.append(output.detach().cpu().numpy())
            targets.append(target.detach().squeeze(-1).cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    train_rmse_loss = np.sqrt(mean_squared_error(predictions, targets))
    return train_rmse_loss


def validate_fn(val_loader, model):
    model.eval()

    predictions = []
    targets = []
    for batch in val_loader:
        with torch.no_grad():
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)

            output = model(ids, mask)
            output = output["logits"].squeeze(-1)

            target = batch["targets"].to(device)

            predictions.append(output.detach().cpu().numpy())
            targets.append(target.detach().squeeze(-1).cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    valid_rmse_loss = np.sqrt(mean_squared_error(predictions, targets))

    return valid_rmse_loss


def train(model, train_loader, val_loader, optimizer):
    train_scores = []
    val_scores = []
    best_score = np.inf
    for epoch in tqdm(range(cfg.EPOCHS)):
        print("---------------" + str(epoch) + "start-------------")

        train_rmse = train_fn(train_loader, model, optimizer)
        train_scores.append(train_rmse)

        val_rmse = validate_fn(val_loader, model)
        val_scores.append(val_rmse)

        print(f'Train loss: {train_rmse}, val loss: {val_rmse}')

        if best_score > val_rmse:
            best_score = val_rmse

            state = {
                'state_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                "bestscore": best_score
            }

            torch.save(state, cfg.MODEL_SAVE_PATH)
    return train_scores, val_scores


def main():
    df = pd.read_csv(cfg.SPLIT_DATAFRAME)

    tokenizer = transformers.BertTokenizer.from_pretrained(cfg.BERT_PATH)

    train_df, val_df = train_test_split(df, random_state=cfg.SEED, test_size=0.2)

    train_ds = BERTDataSet(train_df['text_preprocessed'].values, train_df['target'].values, tokenizer)
    val_ds = BERTDataSet(val_df['text_preprocessed'].values, val_df['target'].values, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)

    model = transformers.BertForSequenceClassification.from_pretrained(cfg.BERT_PATH, num_labels=1)
    model.to(device)

    lr = 2e-5
    optimizer = AdamW(model.parameters(), lr, betas=(0.9, 0.999), weight_decay=1e-2)
    train_scores, val_scores = train(model, train_loader, val_loader, optimizer)

    print(train_scores)
    print(val_scores)


if __name__ == '__main__':
    main()
