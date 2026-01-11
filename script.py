import os
import cv2

def load_images(directory: str):
    images = []
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)

        if image is not None:
            images.append(image)
            print(f"Loaded image: {filename} with shape {image.shape}")
        else:
            print(f"Failed to load image: {filename}")
    
    return images


def load_images_recursive(directory: str):
    images = []
    for root, _, files in os.walk(directory):
        for filename in files:
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)

            if image is not None:
                images.append(image)
                print(f"Loaded image: {image_path} with shape {image.shape}")
            else:
                print(f"Failed to load image: {image_path}")
    
    return images


if __name__ == "__main__":
    input_folder = './input'
    load_images_recursive(input_folder)