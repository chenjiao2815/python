from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
from config import *
import os
import uuid
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import json
from datetime import datetime
import glob
from typing import List, Optional
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建保存上传图片和结果的目录
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
HISTORY_FILE = "history.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 初始化历史记录文件
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

def save_to_history(result_base64):
    """保存结果到历史记录"""
    try:
        # 读取现有历史记录
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # 添加新记录
        history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'result': result_base64
        })
        
        # 只保留最近50条记录
        if len(history) > 50:
            history = history[-50:]
        
        # 保存更新后的历史记录
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存历史记录时出错: {e}")

# 加载模型
device = torch.device('cpu')
model = get_model()
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth'), map_location=device))
model = model.to(device)
model.eval()

def load_image(image):
    """加载并预处理图像"""
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
    return image_tensor.unsqueeze(0), original_size

def predict(image_tensor):
    """使用模型进行预测"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output)
    return output

def postprocess(output, original_size, threshold=0.5):
    """后处理预测结果"""
    pred_mask = output.squeeze().cpu().numpy()
    pred_mask = (pred_mask > threshold).astype(np.uint8)
    pred_mask = Image.fromarray(pred_mask)
    pred_mask = pred_mask.resize(original_size, Image.NEAREST)
    return pred_mask

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # 生成唯一文件名
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_input.jpg")
    result_path = os.path.join(RESULT_DIR, f"{file_id}_result.png")
    
    # 保存上传的图片
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)
    
    # 处理图片
    image = Image.open(BytesIO(contents)).convert('RGB')
    image_tensor, original_size = load_image(image)
    output = predict(image_tensor)
    pred_mask = postprocess(output, original_size)
    
    # 创建可视化结果
    plt.figure(figsize=(12, 6))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(image))
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示预测掩码
    plt.subplot(1, 2, 2)
    plt.imshow(np.array(pred_mask), cmap='gray')
    plt.title('预测的道路掩码')
    plt.axis('off')
    
    # 保存结果
    plt.savefig(result_path)
    plt.close()
    
    # 将结果转换为base64
    with open(result_path, "rb") as f:
        result_base64 = base64.b64encode(f.read()).decode()
    
    # 保存到历史记录
    save_to_history(result_base64)
    
    return {"result": result_base64}

@app.get("/history")
async def get_history():
    """获取历史记录"""
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        return history
    except Exception as e:
        print(f"读取历史记录时出错: {e}")
        return []

@app.get("/dataset/info")
async def get_dataset_info():
    """获取数据集信息"""
    try:
        # 获取图像文件列表
        image_files = glob.glob(os.path.join(DATA_DIR, 'imgs', '*.jpg'))
        total_images = len(image_files)
        
        # 获取第一张图片的尺寸
        if total_images > 0:
            with Image.open(image_files[0]) as img:
                image_width, image_height = img.size
        else:
            image_width, image_height = 0, 0
        
        # 计算数据集总大小
        total_size = 0
        for img_file in image_files:
            total_size += os.path.getsize(img_file)
            mask_file = img_file.replace('imgs', 'mask').replace('.jpg', '.png')
            if os.path.exists(mask_file):
                total_size += os.path.getsize(mask_file)
        
        return {
            "total_images": total_images,
            "image_width": image_width,
            "image_height": image_height,
            "total_size": total_size
        }
    except Exception as e:
        print(f"获取数据集信息时出错: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "获取数据集信息失败"}
        )

@app.get("/dataset/preview")
async def get_dataset_preview(
    page: int = Query(1, ge=1),
    per_page: int = Query(9, ge=1, le=50)
):
    """获取数据集预览"""
    try:
        # 获取图像文件列表
        image_files = sorted(glob.glob(os.path.join(DATA_DIR, 'imgs', '*.jpg')))
        total_images = len(image_files)
        
        # 计算分页
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_images)
        current_page_files = image_files[start_idx:end_idx]
        
        # 读取图像和掩码
        preview_data = []
        for img_file in current_page_files:
            # 读取原始图像
            with open(img_file, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode()
            
            # 读取对应的掩码图像
            mask_file = img_file.replace('imgs', 'mask').replace('.jpg', '.png')
            with open(mask_file, 'rb') as f:
                mask_base64 = base64.b64encode(f.read()).decode()
            
            # 获取文件名（不含路径和扩展名）
            filename = os.path.splitext(os.path.basename(img_file))[0]
            
            preview_data.append({
                "filename": filename,
                "image": image_base64,
                "mask": mask_base64
            })
        
        return {
            "total": total_images,
            "page": page,
            "per_page": per_page,
            "images": preview_data
        }
    except Exception as e:
        print(f"获取数据集预览时出错: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "获取数据集预览失败"}
        )

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html") 