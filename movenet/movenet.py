import torch
import torch.nn as nn
from torch.nn import functional as F

def affinemodule(a, b, tx, ty):
    return torch.Tensor([[a,-b,tx],[b,a,ty]])


class MoveNet(nn.Module):
    def __init__(self, layers_num:int, H:int, W:int):
        self.layers_num = layers_num
        self.affines= []
        self.H = torch.Tensor(H, requires_grad=False)
        self.W = torch.Tensor(W, requires_grad=False)
        for _ in range(layers_num-1):
            alpha = torch.Tensor(1., requires_grad=True)
            beta = torch.Tensor(1., requires_grad=True)
            tx = torch.Tensor(0., requires_grad=True)
            ty = torch.Tensor(0., requires_grad=True)
            a = torch.exp(torch.sigmoid(alpha))
            b = torch.exp(torch.sigmoid(beta))
            x = torch.sigmoid(tx) * self.W
            y = torch.sigmoid(ty) * self.H
            self.affines.append(affinemodule(a, b, x, y))
    def image_add(self, back, front, mask):
        mask_inverse = torch.bitwise_not(mask)
        back =  torch.bitwise_and(back, mask_inverse)
        return back+front
    
    def forward(self, background, layers_images, layers_mask_images):
        grid = F.affine_grid(self.affine, layers_mask_images.size())
        layers_mask_images = F.grid_sample(layers_mask_images, grid)
        for mask_image, mask in zip(layers_images, layers_mask_images):
            background = self.image_add(background, mask_image, mask)
        return background
    

if __name__ == '__main__':
    net = MoveNet(1, 480, 640)
    