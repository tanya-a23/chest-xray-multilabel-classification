# Train a CNN for image classification

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChestXrayDataset
from model import XrayClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import time


# Performance metric: macro accuracy
def macro_accuracy(outputs, targets, threshold=0.5):
    preds = (outputs >= threshold).float()
    correct = (preds == targets).float()
    acc_per_label = correct.mean(dim=0)
    macro_acc = acc_per_label.mean()
    return macro_acc.item()


# Initialization parameters
# Experiment number
EXP = 1

CSV_PATH = 'data/Data_Entry_train_val_2.csv'
IMG_DIR = 'data/'

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 7
LR = 1e-4

# TensorBoard
writer = SummaryWriter(log_dir = f'runs/exp_{EXP}')

# Use GPU, if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Dataset
print('Loading dataset...')
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)

train_dataset = ChestXrayDataset('data/train.csv', IMG_DIR)
val_dataset = ChestXrayDataset('data/val.csv', IMG_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = XrayClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Train the model
print('Training...')
for epoch in range(EPOCHS):
    start = time.time()

    print(f'Epoch {epoch} / {EPOCHS-1}')

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train() # Set the model in training mode
        else:
            model.eval() # Set the model in validation mode

        running_loss = 0
        running_macro_acc = 0
        steps = 0

        # Iterate over data
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(imgs) # tensor of size BATCH_SIZE*14; values type: float
                loss = criterion(outputs, labels) # loss: tensor; value's type is float
                                                    # labels: tensor of size BATCH_SIZE*14; values are 0. and 1.

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() # float
            batch_acc = macro_accuracy(outputs.sigmoid().detach(), labels)
            running_macro_acc += batch_acc
            steps += 1

            if steps % 100 == 0:
                print(f' Batch {steps}: {phase} loss={loss.item():.4f}, {phase} acc={batch_acc:.4f}')
    
        epoch_loss = running_loss/steps
        epoch_acc = running_macro_acc/steps

        writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
        writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)    

        print(f'Epoch {epoch}/{EPOCHS-1}: {phase} loss: {epoch_loss:.4f}, {phase} acc={epoch_acc:.4f}')

    time_elapsed = time.time() - start
    print(f'Epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


torch.save(model.state_dict(), f'models/model_resnet18_{EXP}.pth')
print('Training complete. Model saved.')