import torch
import copy

class ExponentialMovingAverage:
    """
    实现模型权重的指数移动平均
    """
    def __init__(self, model, decay=0.999):
        """
        初始化EMA
        
        Args:
            model: 要应用EMA的模型
            decay: EMA衰减率 (默认: 0.999)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化影子参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新EMA权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 确保shadow参数在与模型参数相同的设备上
                if self.shadow[name].device != param.data.device:
                    self.shadow[name] = self.shadow[name].to(param.data.device)
                
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用EMA权重到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 确保shadow参数在与模型参数相同的设备上
                if self.shadow[name].device != param.data.device:
                    self.shadow[name] = self.shadow[name].to(param.data.device)
                
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始模型权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
