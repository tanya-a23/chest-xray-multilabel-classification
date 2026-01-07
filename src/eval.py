import torch
from torch.utils.data import DataLoader
from dataset import ChestXrayDataset
from model import XrayClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import time


# Initialization parameters
MODEL_PATH = 'models/model_resnet18_1_20260106.pth'
IMG_DIR = 'data/'
INPUT_FILE = 'data/Data_Entry_test_2.csv'

BATCH_SIZE = 16

# Use GPU, if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Model
model = XrayClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Set the module in evaluation mode

# Dataset
val_dataset = ChestXrayDataset(INPUT_FILE, IMG_DIR)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_labels = []
all_preds = []
start = time.time()

with torch.no_grad():
    # Iterate over data
    for i, (imgs, labels) in enumerate(val_loader):
        if i % 100 == 0:
            print('Batch ', i)

        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs) # logits: [batch_size, 14]
        preds = torch.sigmoid(outputs)  # probabilities

        # Move to CPU
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        
# Concatenate batches
all_labels = np.vstack(all_labels)
all_preds = np.vstack(all_preds)

auc = roc_auc_score(all_labels, all_preds, average='macro')
print("AUC:", auc)

time_elapsed = time.time() - start
print(f'Evaluation complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')