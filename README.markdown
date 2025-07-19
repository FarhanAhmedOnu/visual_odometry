# Visual Odometry with Monocular Camera

## Overview
This project implements a basic visual odometry pipeline for a single camera feed using Python. It processes video frames to detect and match ORB features, estimates camera motion via essential matrix decomposition, and visualizes the camera's trajectory in a separate matplotlib plot. The trajectory is saved as a PNG file (`trajectory.png`).

## Features
- Real-time processing of video feed from a camera or video file.
- ORB feature detection and matching for robust keypoint tracking.
- Camera motion estimation using OpenCV's essential matrix and pose recovery.
- Visualization of the camera trajectory in the X-Z plane.
- Automatic saving of the trajectory plot as a high-resolution PNG.

## Requirements
- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

## Installation
1. Clone or download this repository to your local machine.
2. Install the required dependencies using pip:
   ```bash
   pip install opencv-python numpy matplotlib
   ```
3. Ensure you have a video file (e.g., `1.mp4`) or a working webcam for input.

## Usage
1. **Configure Camera Parameters**:
   - Open `visual_odometry.py`.
   - Adjust the `focal_length` and `principal_point` variables in the script to match your camera's intrinsic parameters. These are critical for accurate motion estimation. Example values are provided:
     ```python
     focal_length = 718.8560  # Example focal length in pixels
     principal_point = (607.1928, 185.2157)  # Example principal point (cx, cy)
     ```

2. **Specify Input**:
   - By default, the script uses a video file named `1.mp4`. Update the video capture line in the `main()` function to use a different file or webcam:
     ```python
     cap = cv2.VideoCapture("1.mp4")  # For video file
     # OR
     cap = cv2.VideoCapture(0)  # For default webcam
     ```

3. **Run the Script**:
   ```bash
   python visual_odometry.py
   ```

4. **Output**:
   - The script displays two windows:
     - **Video Feed**: Shows the input video stream.
     - **Camera Trajectory**: A matplotlib window plotting the camera's X-Z trajectory.
   - The trajectory plot is saved as `trajectory.png` in the working directory, updated with each frame.

5. **Exit**:
   - Press the `q` key while the video feed window is active to stop the script.

## Notes
- **Camera Calibration**: For accurate results, calibrate your camera to obtain the correct `focal_length` and `principal_point`. Default values are examples and may not work for all cameras.
- **Performance**: The script processes frames in real-time but may require optimization for high-resolution videos or low-performance systems.
- **Robustness**: This is a basic implementation. For production use, consider adding outlier rejection, loop closure, or bundle adjustment for improved accuracy.

## File Structure
- `visual_odometry.py`: Main script implementing the visual odometry pipeline.
- `trajectory.png`: Output file containing the latest camera trajectory plot.

## Limitations
- Requires sufficient texture in the scene for reliable ORB feature detection.
- Assumes a calibrated camera; incorrect parameters may lead to inaccurate motion estimation.
- No loop closure or global optimization, so drift may accumulate over time.

## Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Suggestions for enhancing robustness or adding features are welcome!


## Acknowledgments
- Built using OpenCV for computer vision tasks.
- Inspired by standard visual odometry techniques in robotics and computer vision research.
