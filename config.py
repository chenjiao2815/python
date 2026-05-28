import os

# 数据集路径
DATA_DIR = 'data'
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'imgs')
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, 'mask')

# 训练参数
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # Windows 上使用多进程可能会有问题，设为0

# 模型参数
ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'

# 损失函数配置
LOSS = 'dicebce'  # 可选: 'dice', 'bce', 'jaccard', 'focal', 'lovasz', 'tversky', 'combo', 'dicece'
LOSS_WEIGHTS = {
    'dice': 0.7,
    'bce': 0.3,
    'jaccard': 1.0,
    'focal': 1.0,
    'lovasz': 1.0,
    'tversky': 1.0
}
# Tversky损失函数参数
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
# Focal损失函数参数
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# 图像预处理参数
IMAGE_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 保存路径
SAVE_DIR = 'checkpoints'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
