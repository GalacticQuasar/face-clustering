import os
import numpy as np
from sklearn.cluster import DBSCAN
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms

# Initialize MTCNN for face detection and InceptionResnetV1 for embeddings
print("Initializing MTCNN...")
mtcnn = MTCNN(keep_all=True)  # keep_all=True to detect multiple faces

print("Loading InceptionResnetV1...")
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

def get_face_embedding(face_img):
    """
    Applies transformations and extracts embedding from a cropped face image.
    """
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Resize image to 160x160
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(  # Normalize image
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return resnet(transform(face_img).unsqueeze(0)).detach().numpy().flatten()

def get_embeddings(directory: str):
    """
    Walks through the directory, detects faces using MTCNN, and extracts embeddings using InceptionResnetV1.
    Returns:
        embeddings (list): List of embedding vectors.
        image_paths (list): List of corresponding image paths.
    """
    embeddings = []
    image_paths = []

    valid_extensions = {'.jpg', '.jpeg', '.png'}

    print(f"Processing images in '{directory}'...")
    
    for root, _, files in os.walk(directory):
        print("Checking directory:", root)
        for filename in files:
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                image_path = os.path.join(root, filename)
                print("Processing file:", filename)
                
                try:
                    # Load image with PIL
                    img = Image.open(image_path).convert('RGB')
                    
                    # Detect faces in the image
                    boxes, _ = mtcnn.detect(img)
                    
                    if boxes is not None:
                        # Process each detected face
                        for box in boxes:
                            # Crop face from image
                            face_img = img.crop(box)
                            
                            # Extract embedding
                            embedding = get_face_embedding(face_img)
                            embeddings.append(embedding)
                            image_paths.append(image_path)
                    else:
                        print(f"  No faces detected in {filename}")
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    return np.array(embeddings), image_paths

def cluster_faces(embeddings, image_paths, eps=1.0, min_samples=2):
    """
    Clusters face embeddings using DBSCAN with Euclidean Distance.
    """
    if len(embeddings) == 0:
        print("No embeddings found.")
        return

    # DBSCAN with euclidean metric
    # eps values usually range from 0.8 to 1.2 for InceptionResnetV1 embeddings.
    clt = DBSCAN(metric="euclidean", eps=eps, min_samples=min_samples)
    clt.fit(embeddings)

    # Group images by cluster ID
    clusters = {}
    unique_labels = np.unique(clt.labels_)

    for label in unique_labels:
        # Get indices of points in this cluster
        indices = np.where(clt.labels_ == label)[0]
        
        if label == -1:
            cluster_name = "Noise (Outliers)"
        else:
            cluster_name = f"Cluster {label}"
            
        clusters[cluster_name] = [image_paths[i] for i in indices]

    return clusters

if __name__ == "__main__":
    # 1. Configuration
    INPUT_DIR = "./input"  # Change this to your folder path
    
    # 2. Get Embeddings using MTCNN and InceptionResnetV1
    embeddings_np, paths = get_embeddings(INPUT_DIR)
    
    print(f"\nExtracted {len(embeddings_np)} faces.")

    # Save embeddings into file temporarily for debugging
    np.save("embeddings.npy", embeddings_np)

    # Pull embeddings from file to ensure they are correctly saved and loaded
    loaded_embeddings = np.load("embeddings.npy")
    embeddings_np = loaded_embeddings

    # 3. Run Clustering
    # Using euclidean metric to match the distance calculation in old facial recognition code.
    # eps around 0.8-1.2 works well for InceptionResnetV1 embeddings with euclidean distance.
    results = cluster_faces(embeddings_np, paths, eps=1.0, min_samples=2)

    # 4. Print Results
    print("\n--- Clustering Results ---")
    for cluster_name, images in results.items():
        print(f"\n{cluster_name}: {len(images)} faces")
        for img in images[:5]:  # Print first 5 images per cluster
            print(f" - {img}")
        if len(images) > 5:
            print(f"   ... and {len(images) - 5} more")
