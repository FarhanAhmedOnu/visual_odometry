import cv2
import numpy as np
import matplotlib.pyplot as plt

# Camera parameters (assumed calibrated, adjust as needed)
focal_length = 718.8560  # Example focal length in pixels
principal_point = (607.1928, 185.2157)  # Example principal point (cx, cy)
K = np.array([[focal_length, 0, principal_point[0]],
              [0, focal_length, principal_point[1]],
              [0, 0, 1]], dtype=np.float32)

def extract_features(image):
    """Extract ORB features and descriptors from an image."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """Match features between two frames using BFMatcher."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def estimate_motion(matches, kp1, kp2, K):
    """Estimate camera motion using essential matrix decomposition."""
    # Extract matched points
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Compute essential matrix
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover pose
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    
    return R, t, mask

def update_trajectory_plot(poses, fig, ax):
    """Update and save the trajectory plot."""
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Camera Trajectory')
    ax.grid(True)
    
    # Plot trajectory
    x = [p[0] for p in poses]
    z = [p[2] for p in poses]
    ax.plot(x, z, 'b-', marker='o', markersize=3)
    
    # Save the plot
    plt.savefig('trajectory.png', dpi=300, bbox_inches='tight')
    plt.pause(0.01)  # Brief pause to update plot

def main():
    # Initialize video capture (0 for default camera, or specify video file path)
    cap = cv2.VideoCapture("1.mp4")
    if not cap.isOpened():
        print("Error: Could not open video feed.")
        return
    
    # Initialize pose
    trajectory = np.eye(4)  # 4x4 transformation matrix
    poses = [trajectory[:3, 3]]  # Store camera positions
    
    # Initialize matplotlib plot
    plt.ion()  # Interactive mode for real-time updates
    fig, ax = plt.subplots()
    
    prev_image = None
    prev_kp = None
    prev_desc = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        kp, desc = extract_features(gray)
        
        if prev_image is not None:
            # Match features with previous frame
            matches = match_features(prev_desc, desc)
            
            if len(matches) > 10:  # Minimum number of matches
                # Estimate motion
                R, t, mask = estimate_motion(matches, prev_kp, kp, K)
                
                # Update pose
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.ravel()
                trajectory = trajectory @ T
                
                # Store new position
                poses.append(trajectory[:3, 3])
                
                # Update trajectory plot
                update_trajectory_plot(poses, fig, ax)
                
                # Show video feed
                cv2.imshow('Video Feed', frame)
        
        # Update previous frame data
        prev_image = gray
        prev_kp = kp
        prev_desc = desc
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    plt.close()

if __name__ == "__main__":
    main()