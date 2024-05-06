from run_inference import inference_single_image
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

def prepare_model_input(image_name, masks_dir, images_dir):
    ref_mask_path = os.path.join(masks_dir, image_name)
    ref_image_path = os.path.join(images_dir, image_name)

    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY)

    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_mask = (ref_mask > 0).astype(np.uint8)
    tar_mask = ref_mask.copy()
    tar_image = ref_image.copy()

    return ref_image, ref_mask, tar_image, tar_mask

def generate_images(ref_image, ref_mask, tar_image, tar_mask):
    gen_images = []
    for i in range(10):
        random.seed(random.randint(0, 10000))
        gen_image = inference_single_image(ref_image, ref_mask, tar_image, tar_mask)
        # gen_path = os.path.join(save_dir, image_name)
        gen_images.append(gen_image)
    return gen_images

def save_images(save_dir, images):
    os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(images):
        save_path = os.path.join(save_dir, f"image{i}.png")
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    mask_dir = "/data/om/reflection_anydoor/dataset/train/masks"
    images_dir ="/data/om/reflection_anydoor/dataset/train/images"
    gen_dir = "/data/om/reflection_anydoor/dataset/train/generated_images"
    for file in os.listdir(images_dir):
        base, _ = os.path.splitext(file)
        save_dir = os.path.join(gen_dir, base)
        ref_image, ref_mask, tar_image, tar_mask = prepare_model_input(
            image_name=file, 
            masks_dir=mask_dir, 
            images_dir=images_dir)

        gen_images = generate_images(ref_image, ref_mask, tar_image, tar_mask)
        save_images(save_dir, gen_images)
