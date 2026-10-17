from fileinput import filename
import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_image_paths(directory: str):
    """
    Walks through the directory and collects paths of valid image files.
    Returns:
        image_paths (list): List of image file paths.
    """
    image_paths = []
    valid_extensions = {'.jpg', '.jpeg', '.png'}

    print(f"Scanning directory '{directory}' for images...")
    
    for root, _, files in os.walk(directory):
        print("Checking directory:", root)
        for filename in files:
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                image_path = os.path.join(root, filename)
                image_paths.append(image_path)
                print("Found image file:", filename)

    return image_paths

def get_embeddings(image_path: str, model_name: str = "Facenet512", max_side: int = 1920):
    """
    Detects faces in the image and extracts embeddings using DeepFace.

    Returns:
        embedding (list): Embedding vectors for each face in the image.
    """
    embeddings = []

    valid_extensions = {'.jpg', '.jpeg', '.png'}

    filename = os.path.basename(image_path)
    print("Processing file:", filename)
    
    try:
        # Downscale large photos so detection doesn't run out of memory
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("could not read image")
        scale = max_side / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # DeepFace.represent returns a list of dicts (one for each face detected)
        representation = DeepFace.represent(
            img_path=img,
            model_name=model_name,
            enforce_detection=False
        )
        
        # Store each face found in the image
        for face_obj in representation:
            embeddings.append(face_obj['embedding'])
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    return embeddings

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

def visualize_embeddings(embeddings, labels=None, method='tsne', title='Face Embeddings Visualization'):
    """
    Visualizes embeddings in 2D using dimensionality reduction.
    
    Args:
        embeddings (np.array): Array of embeddings (n_samples, n_features)
        labels (np.array): Cluster labels for each embedding (optional)
        method (str): Dimensionality reduction method - 'tsne', 'pca', or 'both'
        title (str): Title for the plot
    """
    if len(embeddings) == 0:
        print("No embeddings to visualize.")
        return
    
    # Normalize embeddings if they aren't already
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Set up labels if not provided
    if labels is None:
        labels = np.zeros(len(embeddings))
    
    # Determine number of unique clusters (excluding noise if -1 is present)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    n_noise = np.sum(labels == -1)
    
    print(f"Visualizing {len(embeddings)} embeddings with {n_clusters} clusters and {n_noise} noise points...")
    
    if method == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        axes = [ax1, ax2]
        methods = ['pca', 'tsne']
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        axes = [ax]
        methods = [method]
    
    for ax, current_method in zip(axes, methods):
        # Apply dimensionality reduction
        if current_method == 'tsne':
            print("Applying t-SNE reduction...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = reducer.fit_transform(embeddings_normalized)
            method_title = "t-SNE"
        elif current_method == 'pca':
            print("Applying PCA reduction...")
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings_normalized)
            method_title = f"PCA (explained variance: {reducer.explained_variance_ratio_.sum():.2%})"
        else:
            raise ValueError("Method must be 'tsne', 'pca', or 'both'")
        
        # Create color map
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for label_idx, label in enumerate(unique_labels):
            mask = labels == label
            
            if label == -1:
                # Noise points
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c='black', marker='x', s=50, alpha=0.5,
                          label='Noise')
            else:
                # Regular cluster points
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=[colors[label_idx]], marker='o', s=100, alpha=0.7,
                          edgecolors='black', linewidths=0.5,
                          label=f'Cluster {label}')
        
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(f'{title}\n({method_title})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embeddings_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'embeddings_visualization.png'")
    plt.show()

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "./input"  # Change this to your folder path
    LOAD_FROM_DISK = False  # Set to True to skip embedding generation and load from disk
    
    if LOAD_FROM_DISK:
        # Load previously saved embeddings and paths
        print("Loading embeddings from disk...")
        face_image_paths = np.load("face_image_paths.npy", allow_pickle=True)
        embeddings = np.load("embeddings.npy")
        print(f"Loaded {len(embeddings)} embeddings from disk.")
    else:
        # Get image paths
        image_paths = get_image_paths(INPUT_DIR)
        
        # Get Embeddings
        # 'Facenet512' is recommended for clustering as it produces 512d vectors.
        embeddings = []
        face_image_paths = []  # Track which image each embedding came from
        
        for image_path in image_paths:
            face_embeddings = get_embeddings(image_path, model_name="VGG-Face")
            embeddings.extend(face_embeddings)
            # Store the source image path for each face embedding
            face_image_paths.extend([image_path] * len(face_embeddings))
        
        # Convert to numpy arrays
        embeddings = np.array(embeddings)
        face_image_paths = np.array(face_image_paths)
        
        # Save to disk for future use
        print(f"\nSaving {len(embeddings)} embeddings to disk...")
        np.save("face_image_paths.npy", face_image_paths)
        np.save("embeddings.npy", embeddings)
        print("Saved successfully!")

    # Run Clustering
    # eps=0.4 is a good starting point for Cosine distance.
    # Lower eps = stricter (more clusters, more noise). Higher eps = looser (fewer clusters).
    print(f"\nClustering {len(embeddings)} face embeddings...")
    results = cluster_faces(embeddings, face_image_paths, eps=0.4, min_samples=3)

    # Print Results
    print("\n--- Clustering Results ---")
    for cluster_name, images in results.items():
        print(f"\n{cluster_name}: {len(images)} faces")
        for img in images[:5]:  # Print first 5 images per cluster
            print(f" - {img}")
        if len(images) > 5:
            print(f"   ... and {len(images) - 5} more")

    # Save full clustering results to a text file
    with open("clustering_results.txt", "w") as f:
        for cluster_name, images in results.items():
            f.write(f"{cluster_name}: {len(images)} faces\n")
            for img in images:
                f.write(f" - {img}\n")
            f.write("\n")
    
    # Visualize embeddings
    # Extract cluster labels from DBSCAN results
    clt = DBSCAN(metric="cosine", eps=0.4, min_samples=3)
    cluster_labels = clt.fit_predict(embeddings)
    
    # Visualize with both methods
    visualize_embeddings(embeddings, labels=cluster_labels, method='both', 
                        title='Face Clustering Results')
