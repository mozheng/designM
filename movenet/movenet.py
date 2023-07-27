import torch
import torch.nn as nn
from torch.nn import functional as F


class MoveNet(nn.Module):
    def __init__(self, layers_num:int, H:int, W:int):
        super().__init__()
        assert layers_num > 1
        self.layers_num = layers_num
        self.H = torch.tensor(H, requires_grad=False)
        self.W = torch.tensor(W, requires_grad=False)
        
        self.affines_param = nn.Parameter(torch.ones(self.layers_num-1 , 4))
            
        self.loss_linear_layer = nn.Linear(6, 1)
        nn.init.xavier_uniform_(self.affines_param)

    def image_add(self, back, front, mask):
        mask_inverse = torch.bitwise_not(mask)
        back =  torch.bitwise_and(back, mask_inverse)
        return back+front
    
    def gen_affine_matrix(self, param):
        alpha = param[:,0]
        beta = param[:,1]
        tx = param[:,2]
        ty = param[:,3]
        a = torch.exp(alpha)
        b = torch.exp(beta)
        tx = tx * self.W
        ty = ty * self.H
        return torch.cat([a,-b,tx, b,a,ty],dim=0).resize(2,3, self.layers_num-1).permute(2,0,1)

    def create_image(self, background, layers_images, layers_mask_images, affine):
        grid = F.affine_grid(affine, layers_mask_images.size())
        background = background.to(torch.uint8)
        layers_mask_images = F.grid_sample(layers_mask_images, grid, align_corners=True).to(torch.uint8)
        layers_images = F.grid_sample(layers_images, grid, align_corners=True).to(torch.uint8)
        for mask_image, mask in zip(layers_images, layers_mask_images):
            background = self.image_add(background, mask_image, mask)
        return background, affine
    
    def forward(self, background, layers_images, layers_mask_images,):
        affine = torch.tanh(self.affines_param)
        affine = self.gen_affine_matrix(affine)
        background, affine = self.create_image(background, layers_images, layers_mask_images, affine)
        output = affine.reshape(self.layers_num-1, -1)
        output = self.loss_linear_layer(output)
        return background, output
    
    def fine_tune(self, index):
        with torch.no_grad():
            self.affines_param[index][2] = self.affines_param[index][2]/2
            self.affines_param[index][3] = self.affines_param[index][3]/2
        

if __name__ == '__main__':
    net = MoveNet(3, 480, 640)
    paras = list(net.parameters())
    for num, para in enumerate(paras):
        print('number:',num)
        print(para)
        print('_____________________________')
    background= torch.zeros(1,3,480,640)
    layers_images=torch.zeros(2,3,480,640)
    layers_mask_images=torch.zeros(2,3,480,640)
    res,output = net()
    print(res.shape)
    