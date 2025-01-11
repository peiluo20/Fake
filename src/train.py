import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from model import DeepFakeCNN
from dataset import DeepFakeDataset
from validate import validate_model
import os

def train_model(train_df, val_df, config):
    train_dataset = DeepFakeDataset(train_df, config)
    ## val_dataset = DeepFakeDataset(val_df, config)
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    ## val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFakeCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.4f}")
        ## validate_model(model, val_loader, criterion)

    # 创建保存目录
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    torch.save(model.state_dict(), f"{config['training']['checkpoint_dir']}/model.pth")
