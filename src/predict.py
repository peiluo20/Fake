import os
from model import DeepFakeCNN
from torchvision import transforms
from PIL import Image
import torch

# 加载模型函数
def load_model(model_path, device):
    model = DeepFakeCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 单张图片预测函数
def predict(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = (output.item() > 0.5)
    return "AI合成" if prediction else "非AI合成"

# 批量预测函数
def batch_predict(image_dir, model, device):
    results = []
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # 跳过非图片文件
        try:
            result = predict(image_path, model, device)
            results.append((image_file, result))
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results.append((image_file, "Error"))
    return results

if __name__ == "__main__":
    # 图片目录和模型路径
    image_dir = "../data/faces_225"  # 图片目录
    model_path = "../saved_models/model.pth"  # 模型文件路径

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_model(model_path, device)

    # 批量预测
    results = batch_predict(image_dir, model, device)

    # 打印结果
    print("批量预测结果：")
    for image_file, result in results:
        print(f"{image_file}: {result}")

    # 保存结果到文件（可选）
    with open("batch_prediction_results.txt", "w") as f:
        for image_file, result in results:
            f.write(f"{image_file}: {result}\n")
