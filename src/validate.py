import torch

def validate_model(model, val_loader, criterion, device="cuda"):
    """
    验证模型性能的函数
    Args:
        model: 已训练的 PyTorch 模型
        val_loader: 验证数据的 DataLoader
        criterion: 损失函数
        device: 使用的设备（默认 'cuda'）
    """
    model.eval()  # 切换到评估模式
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in val_loader:
            # 将数据传输到设备
            inputs, labels = inputs.to(device), labels.float().to(device)

            # 模型预测
            outputs = model(inputs).squeeze()

            # 计算损失
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算预测的准确性
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.4f}")
    return val_loss / len(val_loader), accuracy
