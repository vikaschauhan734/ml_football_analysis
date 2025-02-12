## ML Football Analysis

Dataset link: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1

## Overview
This project performs advanced football analysis using **YOLO-based object detection**, **optical flow**, and **perspective transformation**. It tracks players, referees, and the ball, assigns teams based on jersey color, measures ball possession, tracks player movement in real-world metrics, and calculates speed and distance covered.

## Features
- **Player, referee, and ball detection** using YOLO.
- **Team assignment** based on jersey color classification.
- **Ball possession analysis** to measure control percentage.
- **Optical flow-based camera movement estimation** for precise tracking.
- **Perspective transformation** for real-world distance measurement.
- **Player tracking** across frames using bounding box trajectory.
- **Speed and distance estimation** for performance insights.

## Installation
### Prerequisites
- Python 3.x
- YOLO (Ultralytics)
- OpenCV
- TensorFlow/PyTorch
- Roboflow
- Flask (if deploying as a web application)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/football-analysis.git
   cd football-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from **Roboflow** and place it in the expected YOLO format:
   ```
   football-players-detection.v1i.yolov11/football-players-detection.v1i.yolov11
   ```
4. Train the YOLO model on **Google Colab**:
   - Upload the dataset
   - Load `yolo11n.pt`
   - Start training
   - Download the trained model (`best.pt` and `last.pt`) and place them in the `models/` directory.

## Usage
### 1. **Run YOLO Inference**
   ```bash
   python yolo_inference.py --model models/best.pt --video input.mp4
   ```

### 2. **Run Main Analysis**
   ```bash
   python main.py --video input.mp4
   ```

### 3. **Tracking and Ball Possession**
   - Tracks player movement with bounding box trajectory.
   - Assigns ball possession based on proximity.

### 4. **Camera Movement Estimation**
   - Detects **stable features** in video frames.
   - Adjusts player movement calculations based on camera shifts.

### 5. **Perspective Transformation**
   - Converts **distorted 3D views** into a measurable **2D field**.
   - Computes real-world distance traveled by players.

## Project Structure
```
football-analysis/
│── models/                  # Pretrained YOLO models
│── src/
│   ├── utils/
│   │   ├── video_utils.py   # Video read/write functions
│   │   ├── bbox_utils.py    # Bounding box calculations
│   ├── tracker/
│   │   ├── tracker.py       # Player tracking logic
│   ├── team_assigner/
│   │   ├── team_assigner.py # Assigns teams based on jersey color
│   ├── camera_movement_estimator/
│   │   ├── camera_movement.py # Detects camera motion
│   ├── view_transformer/
│   │   ├── transform.py     # Perspective transformation
│   ├── speed_and_distance_estimator/
│   │   ├── speed_distance.py # Speed and distance calculations
│── main.py                  # Main script to run analysis
│── yolo_inference.py         # Runs YOLO model on video
│── requirements.txt          # List of dependencies
```

## Results & Visualizations
- **Ball possession tracking** displayed via annotation.
- **Player movement paths** visualized using bounding boxes.
- **Camera stabilization adjustments** applied to tracking data.
- **Speed and distance calculations** shown for each player.

## Future Improvements
- **Integrate deep learning models** for event detection.
- **Enhance tracking accuracy** with advanced filtering.
- **Develop a real-time version** for live match analysis.

## License
This project is open-source under the **MIT License**.

## Contact
For questions or contributions, feel free to reach out via GitHub Issues.

