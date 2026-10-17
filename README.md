# Face Clustering

Input: Folder of photos containing faces of multiple people.
Output: Clustered folders of faces of each individual.

## Setup

Requires Python 3.13.

```bash
git clone https://github.com/GalacticQuasar/face-clustering.git
cd face-clustering
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/): `uv venv --python 3.13 && uv pip install -r requirements.txt`

## Input data

Photos are not tracked in git. Put them in an `input/` folder at the repo root (subfolders are fine).

Generated `*.npy` files (cached embeddings) are also gitignored and get regenerated on first run.

## Scripts

- `2-deepface-test.py`: latest experiment. DeepFace embeddings, DBSCAN clustering, PCA/t-SNE plot (`embeddings_visualization.png`). Caches embeddings to `.npy`.
- `deepface-test.py`, `script2.py`: earlier DeepFace experiments.
- `facenet-pytorch-test.py`: facenet-pytorch embeddings + DBSCAN.
- `script.py`: face_recognition-based pipeline (needs the optional `face_recognition` package).
- `old-facial-recognition-test.py`: older webcam-based recognition test.

DeepFace and facenet-pytorch download model weights on first run.
