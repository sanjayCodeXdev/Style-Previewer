# Style Previewer

An AI-powered Augmented Reality (AR) pipeline for eyebrow style recommendation and visualization.

## Features
- **Face Detection & Landmark Tracking**: Uses advanced facial landmarking to identify face shapes and features.
- **Style Recommendation**: Recommends eyebrow styles based on facial geometry.
- **AR Preview**: Real-time visualization of different eyebrow styles on your own face.
- **Face Tilt Compensation**: Dynamically adjusts eyebrow placement based on head movement.

## Project Structure
- `AR_pipeline2.py`: The main AR application logic.
- `GAN_test.py`: Testing script for generative components.
- `subasri_model.pkl`: Pre-trained model for style recommendation.
- `assets/`: Image assets for different eyebrow styles.

## How to Run
1. Ensure you have Python installed.
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe numpy pandas
   ```
3. Run the application:
   ```bash
   python AR_pipeline2.py
   ```

## Author
[sanjayCodeXdev](https://github.com/sanjayCodeXdev)
