Vehicle Audio Damage Detector - YAMNet/Keras demo package

What's included:
- backend/: FastAPI backend that uses a Keras model saved at models/vehicle_classifier.h5
- frontend/: Static frontend (upload + record)
- datasets/: synthetic sample audio for classes: no_issue, engine_knock, brake_squeal, flat_tire, exhaust_leak, gear_noise
- training/: Keras training script (train_keras.py) that creates mel-spectrograms and trains a CNN
- models/: contains vehicle_classifier.h5 (either trained or placeholder)
- preprocessing/: processing utilities
- requirements.txt: Python dependencies for backend and training

Important:
- I attempted to train a Keras model here. See training_log.txt for details.
- YAMNet-based pipeline requires TensorFlow Hub and internet access to download the YAMNet model. I included a Keras mel-spectrogram classifier as a demo that works offline once dependencies are installed.
- To reproduce full YAMNet+classifier training, ensure internet access and modify training script to extract YAMNet embeddings from TensorFlow Hub.

How to run locally:
1. python -m venv venv
2. source venv/bin/activate (Windows: venv\Scripts\activate)
3. pip install -r requirements.txt
4. uvicorn backend.app:app --reload --port 8000
5. Open frontend/index.html and set BACKEND to http://localhost:8000 if needed.

If you'd like, I can now package this directory as a ZIP and provide it for download.
