import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .movenet import MoveNet
from .loss import I2CNLoss

class Trainer:
    def __init__(self,
                 layers_images:list, 
                 layers_mask_images:list,
                 device: str,
                 ) -> None:
        layers_num = len(layers_images)
        assert layers_num >= 2
        assert len(layers_images) + 1 == len(layers_mask_images)
        self.background = layers_images[0]
        self.layers_images = layers_images[1:]
        self.layers_mask_images = layers_mask_images
        _, h, w = self.background.shape
        self.device = device
        self.model = MoveNet(layers_num, h, w).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3).to(self.device)
        self.loss = I2CNLoss(["", ""]).to(self.device)

    def _run_batch(self):
        self.optimizer.zero_grad()
        image= self.model(self.background, self.layers_images, self.layers_mask_images)
        loss = self.loss(image.unsqueeze(0))
        loss.backward()
        self.optimizer.step()
        return image

    def _run_epoch(self, max_iter):
        image = None
        for _ in range(max_iter):
            image = self._run_batch()
        return image
        
    def _save_picture(self, image, imagename):
        image = image.cpu().numpy()
        Image.fromarray(image.astype(np.uint8)).save(imagename)
        

    def train(self, max_epochs: int, max_iter):
        for epoch in range(max_epochs):
            image = self._run_epoch(max_iter)
            self._save_picture(image, "image_{}.jpg".format(epoch))



def main(layers_images, layers_mask_images):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(layers_images, layers_mask_images, device)
    trainer.train(1, 50)


if __name__ == "__main__":
    
    
    main(layers_images, layers_mask_images)
