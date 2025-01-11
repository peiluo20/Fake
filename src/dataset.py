import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DeepFakeDataset(Dataset):
    def __init__(self, dataframe, config):
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.Resize(config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        label = 1 if row['label'] == 'FAKE' else 0
        return self.transform(image), label
