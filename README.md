# 🎨 AI SketchVision — Smart Drawing Board

**Draw with your hand. Let AI recognize your sketch and display a real image!**

AI SketchVision is an AI-powered, gesture-controlled drawing application built with Python, OpenCV, MediaPipe, and TensorFlow. It allows users to draw in the air using hand gestures, classify their sketches using a Convolutional Neural Network (CNN), and display a corresponding real image from a local image library.

## ✨ Features

- 🖐️ **Hand Gesture Tracking:** Draw using your index finger in front of your webcam.
- 🎨 **Interactive Drawing Board:** Create sketches using hand movements.
- 🧠 **AI Sketch Classification:** Classifies drawings into five categories using a trained CNN model.
- 🖼️ **Real Image Display:** Displays a corresponding local image when the prediction confidence reaches the configured threshold.
- 🧹 **Eraser and Clear:** Erase strokes or clear the drawing board.
- ↩️ **Undo:** Undo previous drawing actions.
- 🎯 **Prediction Confidence:** Displays the predicted class and confidence score.
- ⚡ **Real-Time Interaction:** Combines computer vision and machine learning for an interactive experience.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Webcam access, image processing, and drawing interface |
| MediaPipe | Hand landmark detection and gesture tracking |
| TensorFlow / Keras | CNN model loading and sketch classification |
| NumPy | Numerical and image-array operations |

## 🧠 Supported Sketch Categories

The trained model currently recognizes these five categories:

- 🍎 Apple
- 🚗 Car
- 🏠 House
- ⭐ Star
- 🌳 Tree

The application uses the model stored at `models/sketch_classifier.keras` and class labels from `models/class_names.json`.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/chiranjibsahoo29/AI_SketchVision.git
cd AI_SketchVision
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

If the repository includes `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages:

```bash
pip install opencv-python mediapipe tensorflow numpy
```

*Note: MediaPipe and TensorFlow compatibility depends on your Python version and operating system.*

### 4. Verify the project files

Make sure the following files and folders exist:

```text
AI_SketchVision/
├── assets/
│   └── real_images/
│       ├── apple.jpg
│       ├── car.jpg
│       ├── house.jpg
│       ├── star.jpg
│       └── tree.jpg
├── models/
│   ├── class_names.json
│   └── sketch_classifier.keras
├── drawing.py
├── tracking.py
├── main.py
├── hand_landmarker.task
├── requirements.txt
└── README.md
```

The actual project structure may contain additional files.

### 5. Run the application

```bash
python3 main.py
```

Allow webcam access if your operating system requests permission.

## 🎮 Keyboard Shortcuts

| Key | Action |
|---|---|
| `C` | Clear the drawing board |
| `E` | Toggle eraser mode |
| `U` | Undo the last drawing action |
| `Q` | Quit the application |

## 🔄 How It Works

1. **Capture:** OpenCV captures live video from the webcam.
2. **Track:** MediaPipe detects hand landmarks and tracks finger movements.
3. **Draw:** The application converts the tracked index-finger movement into strokes on the virtual drawing board.
4. **Classify:** After drawing pauses, the sketch is processed and passed to the trained CNN model.
5. **Predict:** The model predicts the sketch category and confidence score.
6. **Display:** If the prediction meets the configured confidence threshold, the application attempts to display the corresponding image from `assets/real_images/`.

## 📁 Project Structure

- `main.py` — Application entry point, webcam interface, prediction workflow, and image display.
- `tracking.py` — Hand tracking and gesture detection.
- `drawing.py` — Drawing board, stroke handling, eraser, colors, and undo functionality.
- `models/sketch_classifier.keras` — Trained CNN classification model.
- `models/class_names.json` — Class labels used by the model.
- `assets/real_images/` — Local reference images for recognized categories.
- `hand_landmarker.task` — Hand landmark model asset used by the configured tracking pipeline.

## 🚀 Future Improvements

- Add more sketch categories.
- Improve classification accuracy for complex and disconnected strokes.
- Generate images from sketches instead of displaying predefined reference images.
- Add a save and export feature for completed drawings.
- Develop a more polished user interface.
- Explore deployment options for a wider range of devices.

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Computer Vision and image processing
- Hand landmark detection and gesture-based interaction
- CNN-based image classification
- Model integration using TensorFlow/Keras
- Real-time webcam application development
- Python project organization and debugging

## 👨‍💻 Author

**Chiranjib Sahoo**

B.Tech Computer Science and Engineering | AI/ML Enthusiast | Python Developer

- GitHub: [@chiranjibsahoo29](https://github.com/chiranjibsahoo29)
- Project Repository: [AI_SketchVision](https://github.com/chiranjibsahoo29/AI_SketchVision)

---

⭐ If you find this project interesting, consider giving the repository a star!

*Built with Python, Computer Vision, and Machine Learning.*
