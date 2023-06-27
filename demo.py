
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamPredictor, sam_model_registry

MODEL_CHECKPOINT = './weights/sam_vit_h_4b8939.pth' # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
MODEL_TYPE = 'vit_h'
DEVICE = 'cpu'


sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT)
sam.to(device=DEVICE)

predictor = SamPredictor(sam_model=sam)
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
)
pipeline = pipeline.to(DEVICE)


def get_mask(image, selected_pixels):  # SelectData is a subclass of EventData
    input_points = np.array(selected_pixels)
    input_labels = np.ones(shape=(input_points.shape[0]))

    image = np.array(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    predictor.set_image(image=image)

    masks, _, _ = predictor.predict(
        point_coords=input_points, 
        point_labels=input_labels, 
        multimask_output=False
    )
    mask = Image.fromarray(masks[0, :, :])
    return mask
    
def get_inpainting(image, mask, prompt="Blend in with the background"):
    h ,w, _ = image.shape
    image = image.resize((512, 512))
    mask = mask.resize((512, 512))
    image = pipeline(prompt=prompt, image=image, mask_image=mask).images[0]
    image = image.resize((w, h))
    return image


if __name__ == "__main__":
    image = Image.open("images/1.jpg")
    selected_pixels = []
    selected_pixels.append([100, 100])
    mask = get_mask(image, selected_pixels)
    image = get_inpainting(image, mask)
    image.save("a.jpg")
    