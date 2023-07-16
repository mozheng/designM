import torch
import os
import cv2

from movenet.movenet import MoveNet
from movenet.loss import I2CNLoss

class Trainer:
    def __init__(self,
                 background,
                 layers_images, 
                 layers_mask_images,
                 prompts: str,
                 device: str,
                 ) -> None:
        n, c, h, w = layers_images.shape
        self.background = background
        self.layers_images = layers_images
        self.layers_mask_images = layers_mask_images
        self.device = device
        self.prompts = prompts
        self.model = MoveNet(n+1, h, w)
        self.loss = I2CNLoss(self.prompts, self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3).to(self.device)
        self.model.to(self.device)
        self.loss.to(self.device)

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
        image = cv2.cvtColor(image.cpu().numpy(),cv2.COLOR_RGB2BGR)
        cv2.imshow(imagename, image)
        cv2.waitKey()
        

    def train(self, max_epochs: int, max_iter):
        for epoch in range(max_epochs):
            image = self._run_epoch(max_iter)
            self._save_picture(image, "image_{}.jpg".format(epoch))



def main(layers_files:list, masks_files:list):
    assert len(layers_files) > 0
    assert len(layers_files) == len(masks_files) + 1
    layers_images = []
    layers_mask_images = []
    for imagefile in layers_files:
        image = cv2.imread(imagefile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        layers_images.append(image)
    for imagefile in masks_files:
        image = cv2.imread(imagefile)
        layers_mask_images.append(image)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    background = torch.Tensor(layers_images[0]).permute(2,0,1)
    layers_images = torch.Tensor(layers_images[1:]).permute(0,3,1,2)
    layers_mask_images = torch.Tensor(layers_mask_images).permute(0,3,1,2)
    trainer = Trainer(background, layers_images, layers_mask_images,["狗靠在人的腿上",""], device)
    trainer.train(1, 50)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))

    layers_files = [
        os.path.join(project_root, "images", "inpainted_with_mask_dog.png"),
        os.path.join(project_root, "images", "layer_dog.png")
    ]
    masks_files = [os.path.join(project_root, "images", "raw_mask_dog.png")]
    main(layers_files, masks_files)
