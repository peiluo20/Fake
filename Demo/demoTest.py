import pandas as pd
import os

# 加载标签CSV文件
csv_path = "D:/下载/FakeData/archive/metadata.csv"  # 替换为你的CSV文件路径
df = pd.read_csv(csv_path)

# 图片文件夹路径
image_dir = "D:/下载/FakeData/archive/faces_224/"  # 替换为你的图片文件夹路径

# 获取图片文件列表（去除后缀）
image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir)}

# 匹配图片路径
def get_image_path(videoname):
    # 去除后缀，找到匹配的图片
    basename = os.path.splitext(videoname)[0]
    if basename in image_files:
        return os.path.join(image_dir, image_files[basename])
    return None

df['image_path'] = df['videoname'].apply(get_image_path)

# 检查未匹配的图片
missing_images = df[df['image_path'].isnull()]
if not missing_images.empty:
    print("以下视频名没有匹配到图片文件：")
    print(missing_images)
else:
    print("所有图片都成功匹配！")

# 保存结果
df.to_csv("updated_labels.csv", index=False)



from sklearn.model_selection import train_test_split

# 加载更新后的标签数据
df = pd.read_csv("updated_labels.csv")

# 去除没有匹配到图片的行（如果存在）
df = df.dropna(subset=['image_path'])

# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.95, random_state=42, stratify=df['label'])

# 保存划分结果
train_df.to_csv("train_dataset.csv", index=False)
val_df.to_csv("val_dataset.csv", index=False)

print(f"训练集数量: {len(train_df)}, 验证集数量: {len(val_df)}")

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DeepFakeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        label = 1 if row['label'] == 'FAKE' else 0  # 转为数值标签

        # 加载图片
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 统一大小
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 加载训练集和验证集
train_dataset = DeepFakeDataset(pd.read_csv("train_dataset.csv"), transform=transform)

val_dataset = DeepFakeDataset(pd.read_csv("val_dataset.csv"), transform=transform)

# 数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)



import torch.nn as nn
import torch.nn.functional as F

class DeepFakeCNN(nn.Module):
    def __init__(self):
        super(DeepFakeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # 假设输入图片大小为128x128
        self.fc2 = nn.Linear(256, 1)  # 二分类输出
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # 用Sigmoid激活函数
        return x


import torch
import torch.optim as optim

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepFakeCNN().to(device)
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
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

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader):.4f}")
        validate_model(model, val_loader, criterion)


# 验证函数
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {correct / total:.4f}")


train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
torch.save(model.state_dict(), "deepfake_model.pth")


from PIL import Image

def predict(image_path, model):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        prediction = (output.item() > 0.5)
    return "DeepFake" if prediction else "Real"

# 示例
print(predict("D:/下载/FakeData/archive/faces_224/aaagqkcdis.jpg", model))
