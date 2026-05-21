import torch
import torch.nn as nn
from typing import List, Union

class DynamicImbalanceAwareMasking(nn.Module):
    """
    动态不平衡感知遮蔽模块 (Dynamic Imbalance-Aware Masking).
    修改后：使用 ramp_up_epochs 控制难度提升速度，不再依赖 max_epochs。
    """
    def __init__(self,
                 min_p_majority: float,
                 max_p_majority: float,
                 min_p_minority: float,
                 max_p_minority: float,
                 majority_labels: Union[int, List[int]],
                 minority_labels: Union[int, List[int]],
                 ramp_up_epochs: int = 100):  # <--- [修改点] 改名并给默认值
        """
        初始化动态遮蔽模块。
        """
        super().__init__()

        self.min_p_maj = min_p_majority
        self.max_p_maj = max_p_majority
        self.min_p_min = min_p_minority
        self.max_p_min = max_p_minority
        
        # [修改点] 这是一个独立的参数，决定遮蔽率从 min 变到 max 需要多少轮
        # 建议设置为早停轮数的一半左右，例如 50
        self.ramp_up_epochs = max(1, ramp_up_epochs) 
        
        # 初始化状态
        self.current_epoch = 0
        self.current_p_maj = self.min_p_maj
        self.current_p_min = self.min_p_min

        self.majority_labels = set(majority_labels) if isinstance(majority_labels, list) else {majority_labels}
        self.minority_labels = set(minority_labels) if isinstance(minority_labels, list) else {minority_labels}

    def set_epoch(self, epoch: int):
        """
        更新当前的 epoch。
        逻辑：在前 ramp_up_epochs 轮内线性增长，之后保持最大值。
        """
        self.current_epoch = epoch
        
        # [修改点] 核心计算逻辑改变
        # 如果当前 epoch > ramp_up_epochs，进度就锁定为 1.0 (最大难度)
        progress = min(1.0, epoch / self.ramp_up_epochs)
        
        # 动态计算当前遮蔽率
        self.current_p_maj = self.min_p_maj + (self.max_p_maj - self.min_p_maj) * progress
        self.current_p_min = self.min_p_min + (self.max_p_min - self.min_p_min) * progress

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # forward 函数的内容完全不需要动，保持原样即可
        if not self.training:
            return x

        if y.ndim > 1 and y.shape[1] > 1:
            class_indices = torch.argmax(y, dim=1)
        else:
            class_indices = y.view(-1) 

        x_masked = x.clone()

        for i in range(x.shape[0]):
            label = class_indices[i].item()

            if label in self.majority_labels:
                mask_ratio = self.current_p_maj
            elif label in self.minority_labels:
                mask_ratio = self.current_p_min
            else:
                continue 
            
            if mask_ratio <= 1e-6:
                continue

            non_missing_indices = torch.where(~torch.isnan(x[i]))[0]
            if len(non_missing_indices) == 0:
                continue

            num_to_mask = int(len(non_missing_indices) * mask_ratio)
            
            if len(non_missing_indices) > 1:
                num_to_mask = min(num_to_mask, len(non_missing_indices) - 1)
            else:
                num_to_mask = 0
                
            if num_to_mask == 0:
                continue
            
            perm = torch.randperm(len(non_missing_indices), device=x.device)
            indices_to_mask = non_missing_indices[perm[:num_to_mask]]
            
            x_masked[i, indices_to_mask] = float('nan')

        return x_masked