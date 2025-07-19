import cv2
import numpy as np
import threading
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------- Camera Class (from video) --------------------
class Camera:
    def __init__(self, video_path, width=640, height=480):
        self.cap = cv2.VideoCapture(video_path)
        self.width = width
        self.height = height
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file {video_path}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

# -------------------- Frame, KeyFrame, and Map --------------------
class Frame:
    def __init__(self, image, idx):
        self.image = image
        self.idx = idx
        self.Twc = np.eye(4)

class KeyFrame(Frame):
    def __init__(self, frame):
        super().__init__(frame.image, frame.idx)
        self.Twc = frame.Twc.copy()

class Map:
    def __init__(self):
        self.keyframes = []

    def add_keyframe(self, kf):
        self.keyframes.append(kf)
        print(f"[MAP] Added KeyFrame {kf.idx}. Total: {len(self.keyframes)}")

# -------------------- Visual Odometry --------------------
class VisualOdometry:
    def __init__(self, map_obj, camera):
        self.map = map_obj
        self.camera = camera
        self.frame_idx = 0
        self.last_pose = np.eye(4)
        self.poses = []

    def process_next_frame(self):
        ret, frame = self.camera.get_frame()
        if not ret:
            return None, False

        current_frame = Frame(frame, self.frame_idx)
        current_frame.Twc = self.simulate_motion(self.last_pose)
        self.last_pose = current_frame.Twc
        self.poses.append(current_frame.Twc.copy())

        if self.frame_idx % 15 == 0:
            self.map.add_keyframe(KeyFrame(current_frame))

        self.frame_idx += 1
        return current_frame, True

    def simulate_motion(self, last_pose):
        dx = 0.01 + random.uniform(-0.005, 0.005)
        dy = random.uniform(-0.002, 0.002)
        dz = random.uniform(-0.002, 0.002)
        dtheta = random.uniform(-0.01, 0.01)

        translation = np.eye(4)
        translation[0, 3] = dx
        translation[1, 3] = dy
        translation[2, 3] = dz

        rotation = np.eye(4)
        rotation[1, 1] = np.cos(dtheta)
        rotation[1, 2] = -np.sin(dtheta)
        rotation[2, 1] = np.sin(dtheta)
        rotation[2, 2] = np.cos(dtheta)

        return last_pose @ translation @ rotation

# -------------------- Viewer --------------------
class Viewer:
    def __init__(self):
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.display)
        self.thread.start()

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def display(self):
        while self.running:
            with self.lock:
                frame = self.latest_frame
            if frame is not None:
                cv2.imshow("Camera View (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.thread.join()


# -------------------- Plotting --------------------
def plot_trajectory(poses):
    xs, ys, zs = [], [], []
    for Twc in poses:
        x, y, z = Twc[0, 3], Twc[1, 3], Twc[2, 3]
        xs.append(x)
        ys.append(y)
        zs.append(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, marker='o')
    ax.set_title("Simulated Camera Trajectory")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# -------------------- Main --------------------
def main():
    video_path = "1.mp4"  # Replace with your actual video path
    camera = Camera(video_path)
    map_obj = Map()
    vo = VisualOdometry(map_obj, camera)
    viewer = Viewer()

    try:
        while viewer.running:
            frame, success = vo.process_next_frame()
            if not success:
                break
            viewer.update_frame(frame.image)  # Send frame to viewer
            time.sleep(0.033)
    finally:
        viewer.stop()
        camera.release()
        print("[INFO] SLAM pipeline stopped.")

    plot_trajectory(vo.poses)


if __name__ == "__main__":
    main()
