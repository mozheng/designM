 
import torch.nn as nn
import torch
import torch 
from PIL import Image
import numpy as np
import cn_clip.clip as clip
from cn_clip.clip import load_from_name


class ICNMLoss(nn.Module):
    """
    Image-Chinese Match Loss
    """
    def __init__(self, version="ViT-B-16"):

        super(ICNMLoss, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load_from_name(version, device=device, download_root='./')
        self.model.eval()

    def forward(self, image, prompts:list):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(prompts).to(self.device)
        logits_per_image, logits_per_text = self.model.get_similarity(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs


class I2CNLoss(nn.Module):
    """
    Image to Chinese Loss
    """
    def __init__(self, prompts:list, device, version="ViT-B-16"):

        super(I2CNLoss, self).__init__()
        self.device = device
        self.model, self.preprocess = load_from_name(version, device=self.device)
        self.model.eval()
        
        text = clip.tokenize(prompts).to(self.device)
        self.text_features = self.model.encode_text(text)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)  
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image):
        image = self.preprocess(image).to(self.device)
        
        image_features = self.model.encode_image(image)
        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True) 
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ self.text_features.t()
        loss = 1 - logits_per_image.softmax(dim=-1)
        return loss