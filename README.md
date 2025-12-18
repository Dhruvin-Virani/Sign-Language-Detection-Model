# Sign Language Detection Model 🤟

**A small sign-language action recognition project** that uses Mediapipe hand landmarks and a TensorFlow model to detect short sign gestures such as `hello`, `thanks`, and `iloveyou`.

---

## Table of Contents

- **Overview**
- **Project structure**
- **Requirements & Installation**
- **Dataset format**
- **Training / Notebook**
- **Serving (API)**
- **Model & Logs**
- **Adding new actions**
- **Tips / Troubleshooting**
- **Contributing & License**

---

## Overview ✨

This repo contains code and data for training a gesture/phrase classifier using hand landmarks (from Mediapipe). The resulting model (`actions.h5`) can be served via a small FastAPI endpoint (`/predict`).

---

## Project structure 🔧

Key files and folders:

- `main.ipynb` — Notebook with data collection, preprocessing, modeling, training and inference demos.
- `api.py` — Minimal FastAPI server that loads `actions.h5` and exposes `/predict`.
- `actions.h5` — Saved TensorFlow Keras model (artifact produced after training).
- `requirements.txt` — Core Python package list used for development.
- `MP_Data/` — Collected dataset (organized by action/sequence/frame `.npy` files).
- `Logs/` — TensorBoard logs (training runs).

---

## Requirements & Installation ⚙️

Recommended: create a virtual environment.

```bash
pip install -r requirements.txt
# Add server deps (not included in requirements.txt by default):
pip install fastapi uvicorn
```

Notes:
- `tensorflow` and `mediapipe` are used for model training and landmark extraction.
- If running on GPU, install an appropriate `tensorflow` build.

---

## Dataset format (how this project stores data) 📁

Data lives in `MP_Data/` with this layout:

```
MP_Data/
  ├─ hello/
  │   ├─ 0/ (sequence 0)
  │   │   ├─ 0.npy
  │   │   ├─ 1.npy
  │   │   └─ ...
  │   └─ 1/ (sequence 1)
  ├─ thanks/
  └─ iloveyou/
```

- Each frame `.npy` is a 1D array of hand landmarks (21 landmarks × 3 coords = **63 values**).
- `no_sequences = 30` and `sequence_length = 30` are used in the notebook — each sequence is 30 frames long.

---

## Actions / Label mapping 🏷️

The notebook defines the actions and their indices as:

```python
actions = np.array(['hello', 'thanks', 'iloveyou'])
# mapping: 0 -> hello, 1 -> thanks, 2 -> iloveyou
```

The API returns an `action` index (integer) and a `confidence` float. Use the mapping above to convert index → label.

---

## Training / Notebook 🧠

1. Open `main.ipynb` and follow the cells in order.
2. Use the data collection cells to capture sequences (it uses Mediapipe + OpenCV to extract landmarks and saves `.npy` files).
3. Run preprocessing and training cells to create and save `actions.h5`.

Tip: save model with `model.save('actions.h5')` after training and copy it to the repo root for serving.

---

## Serving the model (FastAPI) 🚀

Start the API server:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Example requests:

- curl (replace the array with 63 floats):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [0.0, 0.0, ..., 0.0]}'
```

- Python example:

```python
import requests
resp = requests.post('http://127.0.0.1:8000/predict', json={'landmarks':[0.0]*63})
print(resp.json())  # {"action": 1, "confidence": 0.94}
```

The `api.py` loads `actions.h5` on startup and expects a list of 63 landmark values per example.

---

## Model & Logs 📊

- TensorBoard logs are in `Logs/train`. Launch TensorBoard:

```bash
tensorboard --logdir Logs/train
```

- `actions.h5` is the model used by `api.py` — replace it with a newly trained model as needed.

---

## Adding new actions / extending dataset ➕

1. Create a new folder under `MP_Data/` with the action name.
2. For each sequence create a subfolder `0..N` and save frame `.npy` files named `0.npy, 1.npy, ...` each containing 63 floats (landmark array).
3. Update `actions` in `main.ipynb` (append new label) and re-run training.

---

## Tips / Troubleshooting 💡

- If predictions are poor:
  - Add more sequences and variation (lighting, hand positions).
  - Normalize or augment landmark data.
  - Try different model architectures or longer sequences.
- If the API fails to start, ensure `actions.h5` exists and `fastapi` / `uvicorn` are installed.

---

## Contributing & Contact 🤝

- Open an issue or PR with improvements, bug fixes, or dataset additions.
- Please add a license file (e.g., `LICENSE` with an MIT/Apache license) if you plan to make this public.

---

> Quick note: This README is a starting point — tell me if you want a more detailed usage guide, CI instructions, or a Dockerfile for serving.

---

**Happy prototyping!** 🎉
