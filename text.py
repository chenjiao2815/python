import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
from config import *
import matplotlib.pyplot as plt

def load_image(image_path):
    """加载并预处理图像"""
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 保存原始图像尺寸
    original_size = image.size
    
    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    # 应用预处理
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0), original_size  # 添加batch维度

def predict(model, image_tensor, device):
    """使用模型进行预测"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        # 应用sigmoid激活函数
        output = torch.sigmoid(output)
    return output

def postprocess(output, original_size, threshold=0.5):
    """后处理预测结果"""
    # 将输出转换为numpy数组
    pred_mask = output.squeeze().cpu().numpy()
    
    # 应用阈值
    pred_mask = (pred_mask > threshold).astype(np.uint8)
    
    # 调整回原始图像大小
    pred_mask = Image.fromarray(pred_mask)
    pred_mask = pred_mask.resize(original_size, Image.NEAREST)
    
    return np.array(pred_mask)

def visualize_results(image_path, pred_mask, save_path=None):
    """可视化原始图像和预测结果"""
    # 读取原始图像
    original_image = np.array(Image.open(image_path))
    
    # 创建图像显示
    plt.figure(figsize=(12, 6))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示预测掩码
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('预测的道路掩码')
    plt.axis('off')
    
    # 保存或显示结果
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = get_model()
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    model = model.to(device)
    
    # 设置输入图像路径
    image_path = '23128915_15.jpg'  # 替换为实际的测试图像路径
    
    # 加载和预处理图像
    image_tensor, original_size = load_image(image_path)
    
    # 进行预测
    output = predict(model, image_tensor, device)
    
    # 后处理预测结果
    pred_mask = postprocess(output, original_size)
    
    # 可视化结果
    visualize_results(image_path, pred_mask, save_path='prediction_result.png')

if __name__ == '__main__':
    main()