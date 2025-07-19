"""
Visual Odometry Implementation for Monocular Camera

This script implements a basic visual odometry pipeline using a single camera feed.
It processes video frames to detect and match ORB features, estimates camera motion
using essential matrix decomposition, and visualizes the camera's trajectory in a
separate matplotlib plot. The trajectory is also saved as a PNG file.

Dependencies:
    - OpenCV (cv2): For image processing and feature detection
    - NumPy: For numerical computations and matrix operations
    - Matplotlib: For plotting the camera trajectory

Usage:
    Run the script with a video file (e.g., '1.mp4') or webcam feed.
    Ensure camera parameters are calibrated for accurate results.
    Press 'q' to exit the program.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Camera parameters (assumed calibrated, adjust as needed)
focal_length = 718.8560  # Example focal length in pixels
principal_point = (607.1928, 185.2157)  # Example principal point (cx, cy)
K = np.array([[focal_length, 0, principal_point[0]],
              [0, focal_length, principal_point[1]],
              [0, 0, 1]], dtype=np.float32)  # Camera intrinsic matrix

def extract_features(image):
    """
    Extract ORB features and descriptors from an input image.

    Args:
        image (numpy.ndarray): Grayscale input image.

    Returns:
        tuple: (keypoints, descriptors)
            - keypoints: List of detected ORB keypoints.
            - descriptors: Corresponding ORB descriptors.
    """
    orb = cv2.ORB_create()  # Initialize ORB detector
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    Match ORB features between two frames using brute-force matcher.

    Args:
        desc1 (numpy.ndarray): Descriptors from the first frame.
        desc2 (numpy.ndarray): Descriptors from the second frame.

    Returns:
        list: Sorted list of feature matches based on distance.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Initialize BFMatcher
    matches = bf.match(desc1, desc2)  # Match descriptors
    matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance
    return matches

def estimate_motion(matches, kp1, kp2, K):
    """
    Estimate camera motion between two frames using essential matrix decomposition.

    Args:
        matches (list): List of feature matches between frames.
        kp1 (list): Keypoints from the first frame.
        kp2 (list): Keypoints from the second frame.
        K (numpy.ndarray): Camera intrinsic matrix.

    Returns:
        tuple: (R, t, mask)
            - R (numpy.ndarray): Rotation matrix (3x3).
            - t (numpy.ndarray): Translation vector (3x1).
            - mask (numpy.ndarray): Inlier mask from RANSAC.
    """
    # Extract matched points
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Compute essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover relative pose (rotation and translation)
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    
    return R, t, mask

def update_trajectory_plot(poses, fig, ax):
    """
    Update and save the camera trajectory plot in the X-Z plane.

    Args:
        poses (list): List of 3D camera positions (x, y, z).
        fig (matplotlib.figure.Figure): Matplotlib figure object.
        ax (matplotlib.axes.Axes): Matplotlib axes object for plotting.
    """
    ax.clear()  # Clear previous plot
    ax.set_xlabel('X')  # Set X-axis label
    ax.set_ylabel('Z')  # Set Z-axis label
    ax.set_title('Camera Trajectory')  # Set plot title
    ax.grid(True)  # Enable grid
    
    # Extract X and Z coordinates for plotting
    x = [p[0] for p in poses]
    z = [p[2] for p in poses]
    ax.plot(x, z, 'b-', marker='o', markersize=3)  # Plot trajectory
    
    # Save the plot as a high-resolution PNG
    plt.savefig('trajectory.png', dpi=300, bbox_inches='tight')
    plt.pause(0.01)  # Brief pause to update the plot in real-time

def main():
    """
    Main function to process video feed and perform visual odometry.

    Initializes video capture, processes frames to estimate camera motion,
    updates the trajectory plot, and displays the video feed. The trajectory
    is saved as 'trajectory.png'.
    """
    # Initialize video capture from file '1.mp4'
    cap = cv2.VideoCapture("1.mp4")
    if not cap.isOpened():
        print("Error: Could not open video feed.")
        return
    
    # Initialize camera pose as a 4x4 identity matrix
    trajectory = np.eye(4)
    poses = [trajectory[:3, 3]]  # Store initial camera position
    
    # Initialize matplotlib for real-time trajectory plotting
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()  # Create figure and axes
    
    prev_image = None  # Previous frame (grayscale)
    prev_kp = None  # Previous keypoints
    prev_desc = None  # Previous descriptors
    
    while cap.isOpened():
        ret, frame = cap.read()  # Read next frame
        if not ret:
            break  # Exit if no frame is retrieved
            
        # Convert frame to grayscale for feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract ORB features from current frame
        kp, desc = extract_features(gray)
        
        if prev_image is not None:
            # Match features with previous frame
            matches = match_features(prev_desc, desc)
            
            if len(matches) > 10:  # Ensure enough matches for robust estimation
                # Estimate camera motion
                R, t, mask = estimate_motion(matches, prev_kp, kp, K)
                
                # Update camera pose
                T = np.eye(4)  # Create new transformation matrix
                T[:3, :3] = R  # Set rotation
                T[:3, 3] = t.ravel()  # Set translation
                trajectory = trajectory @ T  # Update cumulative pose
                
                # Store new camera position
                poses.append(trajectory[:3, 3])
                
                # Update and save trajectory plot
                update_trajectory_plot(poses, fig, ax)
                
                # Display current video frame
                cv2.imshow('Video Feed', frame)
        
        # Update previous frame data
        prev_image = gray
        prev_kp = kp
        prev_desc = desc
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()  # Release video capture
    cv2.destroyAllWindows()  # Close OpenCV windows
    plt.close()  # Close matplotlib plot

if __name__ == "__main__":
    main()