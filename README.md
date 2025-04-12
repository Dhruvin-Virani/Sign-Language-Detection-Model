# 🤟 Sign Language Detection Using Computer Vision

This project aims to detect and translate sign language gestures into text using a live camera feed. By leveraging real-time hand and face tracking with machine learning, the model identifies static gestures and converts them into readable text, improving communication accessibility for individuals with hearing or speech impairments.

## 🚀 Features

- Real-time sign detection using webcam
- Hand, face, and pose tracking with **MediaPipe**
- Custom-trained **CNN model** for gesture classification
- Interactive **Jupyter Notebook** environment
- Self-collected dataset with 900 images per sign
- Output sign translation displayed in real time

## 🛠️ Technologies Used

- Python 3.10  
- Jupyter Notebook  
- TensorFlow  
- MediaPipe  
- Protobuf  
- OpenCV  

## 📁 Dataset

A custom dataset was created by capturing 900 frames for each sign. Each image was labeled manually, and **MediaPipe’s landmark detection** was used to extract keypoints. This data was used to train a CNN classifier to distinguish between different sign gestures.

## 🧠 Model Training

- Hand landmarks extracted using MediaPipe
- Normalized and flattened into input arrays
- CNN trained using TensorFlow with categorical cross-entropy loss
- Trained model tested on separate validation data

## 🖼️ Example Output

![Detection Preview](path/to/your/image.png) 

## 📌 How to Run

1. Clone the repository  
2. Install dependencies from `requirements.txt`  
3. Open the Jupyter Notebook  
4. Run all cells to activate the webcam and start detection

```bash
pip install -r requirements.txt
jupyter notebook
```

- Pre-Final Year Project | Code Unnati
