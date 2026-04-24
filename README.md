# ✨ STYLE Previewer: AI-Powered Eyebrow AR Virtual Try-On

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**STYLE Previewer** is a sophisticated Augmented Reality (AR) application that allows users to virtually try on different eyebrow styles in real-time. Using advanced computer vision and machine learning, the system analyzes facial geometry to recommend the most flattering styles tailored to your unique face shape.

---

## 🌟 Key Features

### 🧠 Intelligent AI Recommendation
- **Face Shape Analysis**: Automatically detects face shapes (Oval, Round, Square, Long, Heart) using facial landmark ratios.
- **Eye Spacing Detection**: Analyzes inter-ocular distance to distinguish between wide-set, close-set, and proportional features.
- **Smart Scoring**: Uses a weighted matrix to suggest styles that complement your specific facial structure.

### 🎭 Realistic AR Integration
- **Dynamic Color Tinting**: Samples your natural hair color and automatically tints the virtual eyebrow assets to match.
- **Adaptive Lighting**: Uses histogram matching to ensure virtual assets blend seamlessly with the lighting conditions of your environment.
- **Smooth Landmark Tracking**: Implements custom temporal smoothing to prevent jitter and ensure the virtual eyebrows move naturally with your face.
- **Face Tilt Compensation**: Dynamically adjusts the curvature and position of assets based on head orientation.

### 🎮 Interactive Controls
- **Auto/Manual Modes**: Switch between AI-driven recommendations and manual selection.
- **Real-time Scaling**: Fast adjustment of asset size via on-screen trackbars or keyboard shortcuts.
- **Snapshot Capture**: Save high-quality photos of your new look instantly.

---

## 🛠️ Technical Stack

- **[MediaPipe](https://mediapipe.dev/)**: High-fidelity face mesh tracking with 468+ landmarks.
- **[OpenCV](https://opencv.org/)**: Image processing, real-time rendering, and UI management.
- **[NumPy](https://numpy.org/)**: Heavy-duty mathematical operations for facial geometry analysis.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. 

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sanjayCodeXdev/Style-Previewer.git
   cd Style-Previewer
   ```
2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

### Running the App
Execute the main pipeline script:
```bash
python AR_pipeline2.py
```

---

## ⌨️ Controls

| Action | Control |
| :--- | :--- |
| **Switch Modes** | Press `A` or Click **AUTO/MANUAL** button |
| **Next Style** | Press `S`, `Down Arrow`, or Click **NEXT** |
| **Previous Style** | Press `W`, `Up Arrow`, or Click **PREV** |
| **Scale Up/Down** | Press `+` / `-` or use the **Scale %** trackbar |
| **Take Snapshot** | Click **SHOT** button |
| **Exit App** | Press `ESC` |

---

## 📁 Project Structure

```text
STYLE_Previewer/
├── AR_pipeline2.py     # Main application logic & UI
├── assets/             # Transparent eyebrow style assets
│   └── Eyebrows/       # Stylized eyebrow sheets
├── subasri_model.pkl   # Pre-trained recommendation model
└── README.md           # Project documentation
```

---

## 🤝 Contributors
Developed with ❤️ by:
- **Sanjay Kumar**

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
