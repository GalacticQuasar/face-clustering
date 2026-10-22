import os
import sys
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN

def get_image_paths(directory: str):
    """
    Walks through the directory and collects paths of valid image files.
    Returns:
        image_paths (list): List of image file paths.
    """
    image_paths = []
    valid_extensions = {'.jpg', '.jpeg', '.png'}

    for root, _, files in os.walk(directory):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(root, filename))

    return sorted(image_paths)

def get_faces(image_path: str, model_name: str = "VGG-Face", max_side: int = 1920, padding: float = 0.2):
    """
    Detects faces in the image and extracts an embedding and a cropped image for each.

    Returns:
        faces (list): List of (embedding, crop) tuples, one per detected face.
    """
    faces = []

    print("Processing file:", os.path.basename(image_path))

    try:
        # Downscale large photos so detection doesn't run out of memory
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("could not read image")
        scale = max_side / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        representation = DeepFace.represent(
            img_path=img,
            model_name=model_name,
            enforce_detection=False
        )

        for face_obj in representation:
            # With enforce_detection=False, DeepFace falls back to the whole image
            # with confidence 0 when no face is found. Skip those.
            if face_obj.get('face_confidence', 0) == 0:
                continue

            # Crop the face from the image, with some padding around it
            area = face_obj['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            pad_x, pad_y = int(w * padding), int(h * padding)
            x1, y1 = max(x - pad_x, 0), max(y - pad_y, 0)
            x2, y2 = min(x + w + pad_x, img.shape[1]), min(y + h + pad_y, img.shape[0])

            faces.append((face_obj['embedding'], img[y1:y2, x1:x2]))

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    return faces

def write_clusters(labels, crops, source_paths, output_dir: str):
    """
    Writes each face crop into a folder for its cluster: person_<n> for clusters,
    unknown for DBSCAN noise. Crop filenames include the source photo name.
    """
    for i, (label, crop, source_path) in enumerate(zip(labels, crops, source_paths)):
        folder = "unknown" if label == -1 else f"person_{label}"
        folder_path = os.path.join(output_dir, folder)
        os.makedirs(folder_path, exist_ok=True)

        source_name = os.path.splitext(os.path.basename(source_path))[0]
        cv2.imwrite(os.path.join(folder_path, f"{source_name}_face{i}.jpg"), crop)

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "./input"
    OUTPUT_DIR = "./output"
    MODEL_NAME = "VGG-Face"
    # Lower eps = stricter (more people, more unknowns). Higher eps = looser (fewer people).
    EPS = 0.4
    MIN_SAMPLES = 3

    # Refuse to mix results with a previous run
    if os.path.isdir(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        sys.exit(f"Output folder '{OUTPUT_DIR}' is not empty. Remove it or change OUTPUT_DIR.")

    embeddings = []
    crops = []
    source_paths = []

    for image_path in get_image_paths(INPUT_DIR):
        for embedding, crop in get_faces(image_path, model_name=MODEL_NAME):
            embeddings.append(embedding)
            crops.append(crop)
            source_paths.append(image_path)

    if len(embeddings) == 0:
        sys.exit("No faces found.")

    print(f"\nClustering {len(embeddings)} faces...")
    labels = DBSCAN(metric="cosine", eps=EPS, min_samples=MIN_SAMPLES).fit_predict(np.array(embeddings))

    write_clusters(labels, crops, source_paths, OUTPUT_DIR)

    n_people = len(set(labels) - {-1})
    n_unknown = int(np.sum(labels == -1))
    print(f"Wrote {n_people} person folders and {n_unknown} unknown faces to '{OUTPUT_DIR}'.")
