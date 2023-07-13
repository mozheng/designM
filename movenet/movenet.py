import torch
import torch.nn as nn
from torch.nn import functional as F


class MoveNet(nn.Module):
    def __init__(self, layers_num:int, H:int, W:int):
        super().__init__()
        assert layers_num > 1
        self.layers_num = layers_num
        self.affines= []
        self.H = torch.tensor(H, requires_grad=False)
        self.W = torch.tensor(W, requires_grad=False)
        
        for _ in range(layers_num-1):
            alpha = nn.Parameter(torch.ones(1))
            beta = nn.Parameter(torch.ones(1))
            tx = nn.Parameter(torch.zeros(1))
            ty = nn.Parameter(torch.zeros(1))
            a = torch.exp(torch.sigmoid(alpha))
            b = torch.exp(torch.sigmoid(beta))
            x = torch.sigmoid(tx) * self.W
            y = torch.sigmoid(ty) * self.H

            self.affines.append(nn.ModuleList([a, b, x, y]))
        
    def image_add(self, back, front, mask):
        mask_inverse = torch.bitwise_not(mask)
        back =  torch.bitwise_and(back, mask_inverse)
        return back+front
    
    def affinemodule(self, a, b, tx, ty):
        return torch.cat([a,-b,tx, b,a,ty],dim=0).resize(2,3)

    def forward(self, background, layers_images, layers_mask_images):
        grid = F.affine_grid(self.affine, layers_mask_images.size())
        layers_mask_images = F.grid_sample(layers_mask_images, grid)
        for mask_image, mask in zip(layers_images, layers_mask_images):
            background = self.image_add(background, mask_image, mask)
        return background
    

if __name__ == '__main__':
    net = MoveNet(2, 480, 640)
    paras = list(net.parameters())
    for num, para in enumerate(paras):
        print('number:',num)
        print(para)
        print('_____________________________')
    