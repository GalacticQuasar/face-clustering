import os
from deepface import DeepFace


def get_embeddings(directory: str):
    index = {}  # image path : list of embeddings
    for root, _, files in os.walk(directory):
        for filename in files:
            image_path = os.path.join(root, filename)
            index[image_path] = DeepFace.represent(image_path)


# def cluster


# def visualize


if __name__ == "__main__":
    input_folder = './input'
    index = get_embeddings(input_folder)
