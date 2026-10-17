import os
import cv2
import face_recognition

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


def get_embeddings(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_image, model='hog')
    embeddings = face_recognition.face_encodings(rgb_image, boxes)

    print(f"Found {len(embeddings)} face(s) in the image.")
    return embeddings

def run_db_scan(embeddings, eps=0.5, min_samples=2):
    from sklearn.cluster import DBSCAN
    import numpy as np

    if len(embeddings) == 0:
        print("No embeddings to cluster.")
        return []

    X = np.array(embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"DBSCAN found {num_clusters} clusters.")
    return labels

if __name__ == "__main__":
    input_folder = './input'
    images = load_images_recursive(input_folder)

    all_embeddings = []
    for idx, img in enumerate(images):
        print(f"Processing image {idx + 1}/{len(images)} for embeddings")
        embeddings = get_embeddings(img)
        for i, emb in enumerate(embeddings):
            print(f"Embedding {i + 1}: {emb[:5]}...")  # Print first 5 values of the embedding for brevity
        all_embeddings.extend(embeddings)

    labels = run_db_scan(all_embeddings)
    print(f"Cluster labels: {labels}")
    
    print("Processing completed.")