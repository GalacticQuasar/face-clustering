import os
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN

def get_embeddings(directory: str, model_name: str = "Facenet512"):
    """
    Walks through the directory, detects faces, and extracts embeddings.
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
                    # DeepFace.represent returns a list of dicts (one for each face detected)
                    representation = DeepFace.represent(
                        img_path=image_path,
                        model_name=model_name,
                        enforce_detection=False
                    )
                    
                    # Store each face found in the image
                    for face_obj in representation:
                        embeddings.append(face_obj['embedding'])
                        image_paths.append(image_path)
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    return np.array(embeddings), image_paths

def cluster_faces(embeddings, image_paths, eps=0.4, min_samples=3):
    """
    Clusters face embeddings using DBSCAN with Cosine Distance.
    """
    if len(embeddings) == 0:
        print("No embeddings found.")
        return

    # DBSCAN with cosine metric
    # Note: Scikit-learn's cosine metric is 'distance' (1 - similarity).
    # eps values usually range from 0.3 to 0.5 for normalized face embeddings.
    clt = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples)
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
    
    # 2. Get Embeddings
    # 'Facenet512' is recommended for clustering as it produces 512d vectors.
    embeddings_np, paths = get_embeddings(INPUT_DIR, model_name="OpenFace")
    
    print(f"\nExtracted {len(embeddings_np)} faces.")

    # Save embeddings into file temporarily for debugging
    np.save("embeddings.npy", embeddings_np)

    # Pull embeddings from file to ensure they are correctly saved and loaded
    loaded_embeddings = np.load("embeddings.npy")
    embeddings_np = loaded_embeddings

    # 3. Run Clustering
    # eps=0.4 is a good starting point for Cosine distance.
    # Lower eps = stricter (more clusters, more noise). Higher eps = looser (fewer clusters).
    results = cluster_faces(embeddings_np, paths, eps=0.4, min_samples=3)

    # 4. Print Results
    print("\n--- Clustering Results ---")
    for cluster_name, images in results.items():
        print(f"\n{cluster_name}: {len(images)} faces")
        for img in images[:5]:  # Print first 5 images per cluster
            print(f" - {img}")
        if len(images) > 5:
            print(f"   ... and {len(images) - 5} more")
