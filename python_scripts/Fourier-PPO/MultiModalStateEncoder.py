import torch
import torch.nn as nn

# ===================== 多模态状态编码器 =====================
class MultiModalStateEncoder(nn.Module):
    """
    多模态状态编码器：处理图像和舵机角度
    图像规格：640x480 RGB
    舵机数量：20
    """
    def __init__(self, image_shape=(3, 480, 640), n_servos=20, hidden_dim=512):
        super().__init__()
        
        # 图像编码器（使用ResNet风格）
        self.image_encoder = nn.Sequential(
            # 第一层：640x480 -> 320x240
            nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 320x240 -> 160x120
            
            # 第二层：160x120 -> 80x60
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 80x60 -> 40x30
            
            # 第三层：40x30 -> 20x15
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 20x15 -> 10x7
            
            nn.Flatten(),
            
            # 全连接层
            nn.Linear(128 * 10 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 舵机角度编码器
        self.angle_encoder = nn.Sequential(
            nn.Linear(n_servos, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 时间步编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 融合层
        total_dim = 256 + 32 + 8  # 图像 + 角度 + 时间
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, image, angles, time_step):
        """
        image: [batch, 3, 480, 640]
        angles: [batch, n_servos]
        time_step: [batch, 1] 或标量
        """
        # 图像编码
        image_feat = self.image_encoder(image)
        
        # 角度编码
        angle_feat = self.angle_encoder(angles)
        
        # 时间步编码
        if time_step.dim() == 1:
            time_step = time_step.unsqueeze(-1)
        time_feat = self.time_encoder(time_step)
        
        # 特征融合
        combined = torch.cat([image_feat, angle_feat, time_feat], dim=1)
        return self.fusion(combined)