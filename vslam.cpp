#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <random> // For random numbers in the simplified tracking

// OpenCV for image processing, camera calibration, and basic display
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

// Eigen for efficient linear algebra operations
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Pangolin for lightweight 3D visualization (conceptual inclusion)
// You would need to install Pangolin and link it properly.
// #include <pangolin/pangolin.h> // Uncomment if you have Pangolin installed and configured

// --- Global Mutex for Shared Data (for multi-threading) ---
std::mutex g_data_mutex;

// --- 1. Camera Class ---
// Handles camera parameters and image acquisition.
class Camera {
public:
    cv::VideoCapture cap;
    cv::Mat K; // Intrinsic matrix
    cv::Mat distCoeffs; // Distortion coefficients
    int width, height;

    Camera(int camera_idx = 0, int w = 640, int h = 480) : width(w), height(h) {
        cap.open(camera_idx);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera " << camera_idx << std::endl;
            exit(EXIT_FAILURE);
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

        // Placeholder for camera calibration parameters
        // In a real system, these would be loaded from a file after calibration.
        // Example for a 640x480 camera (adjust as per your calibration)
        K = (cv::Mat_<double>(3, 3) << 
            500.0, 0.0, 320.0,
            0.0, 500.0, 240.0,
            0.0, 0.0, 1.0);
        distCoeffs = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0); // No distortion for simplicity
        
        std::cout << "Camera initialized: " << width << "x" << height << std::endl;
    }

    bool grabFrame(cv::Mat& frame) {
        return cap.read(frame);
    }

    // Undistort image (if distortion coefficients are non-zero)
    void undistort(cv::Mat& image) {
        if (!distCoeffs.empty() && cv::countNonZero(distCoeffs) > 0) {
            cv::undistort(image, image, K, distCoeffs);
        }
    }
};

// --- 2. Frame Class ---
// Stores information for a single camera frame.
class Frame {
public:
    cv::Mat image;
    cv::Mat gray_image;
    double timestamp;
    Eigen::Matrix4d Twc; // Transformation from camera to world (World_T_Camera)
    
    // For direct methods, we might store image gradients or intensity values
    cv::Mat grad_x, grad_y;

    Frame(const cv::Mat& img, double ts, const Eigen::Matrix4d& pose = Eigen::Matrix4d::Identity())
        : image(img), timestamp(ts), Twc(pose) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        // Compute gradients for direct methods (simplified)
        // #pragma omp parallel for // Potential OpenMP parallelization
        cv::Sobel(gray_image, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray_image, grad_y, CV_32F, 0, 1, 3);
    }

    // Copy constructor (needed for storing frames in containers and `last_frame` in VO)
    Frame(const Frame& other) 
        : image(other.image.clone()), 
          gray_image(other.gray_image.clone()),
          timestamp(other.timestamp), 
          Twc(other.Twc),
          grad_x(other.grad_x.clone()),
          grad_y(other.grad_y.clone()) {}

    // Assignment operator
    Frame& operator=(const Frame& other) {
        if (this != &other) {
            image = other.image.clone();
            gray_image = other.gray_image.clone();
            timestamp = other.timestamp;
            Twc = other.Twc;
            grad_x = other.grad_x.clone();
            grad_y = other.grad_y.clone();
        }
        return *this;
    }
};

// --- 3. MapPoint Class ---
// Represents a 3D point in the map.
class MapPoint {
public:
    Eigen::Vector3d position;
    // Other properties like descriptor, normal, etc. could be added
    MapPoint(const Eigen::Vector3d& pos) : position(pos) {}
};

// --- 4. KeyFrame Class ---
// A special frame that is added to the map for optimization.
class KeyFrame : public Frame {
public:
    unsigned long id;
    std::vector<MapPoint*> map_points; // Map points observed in this keyframe (conceptual)

    KeyFrame(const Frame& frame, unsigned long kf_id) : Frame(frame), id(kf_id) {}
};

// --- 5. Map Class ---
// Manages 3D map points and keyframes.
class Map {
public:
    std::vector<MapPoint> map_points;
    std::vector<KeyFrame> keyframes;
    unsigned long next_keyframe_id = 0;

    void addMapPoint(const MapPoint& mp) {
        map_points.push_back(mp);
    }

    void addKeyFrame(const Frame& frame) {
        keyframes.emplace_back(frame, next_keyframe_id++); // Emplace_back constructs in place
        std::cout << "Added KeyFrame " << keyframes.back().id << ". Total Keyframes: " << keyframes.size() << std::endl;
        // In a real system, new map points would be triangulated here
        // and added to the map and associated with the new keyframe.
    }

    // Placeholder for local/global optimization (Bundle Adjustment/Pose Graph)
    void optimizeMap() {
        // This is where Ceres Solver or g2o would be used for non-linear optimization.
        // For CPU optimization:
        // - Use sparse solvers (e.g., Eigen's sparse solvers if implementing custom BA).
        // - Limit iterations to maintain real-time performance.
        // - Potentially run in a separate, lower-priority thread.
        std::cout << "Optimizing map (placeholder)... This is a CPU-intensive task." << std::endl;
        // Example: Simple pose graph optimization (conceptual)
        // for (auto& kf : keyframes) {
        //     kf.Twc = kf.Twc * Eigen::Matrix4d::Identity(); // No actual optimization
        // }
    }
};

// --- 6. VisualOdometry/Tracker Class (Simplified Direct Method) ---
// Estimates camera motion between frames.
class VisualOdometry {
public:
    Map* map;
    Eigen::Matrix4d current_pose; // Current camera pose (World_T_Camera)
    Frame last_frame; // Store last frame directly
    Camera* camera_model;
    bool is_initialized = false;

    // For simplified random motion generation
    std::mt19937 gen; 
    std::uniform_real_distribution<> dis_trans;
    std::uniform_real_distribution<> dis_rot;

    VisualOdometry(Map* m, Camera* cam) 
        : map(m), 
          camera_model(cam),
          gen(std::chrono::system_clock::now().time_since_epoch().count()), // Seed RNG
          dis_trans(-0.01, 0.01), // Small translation range
          dis_rot(-0.005, 0.005)  // Small rotation range (radians)
    {
        current_pose = Eigen::Matrix4d::Identity(); // Initialize at origin
    }

    // Simplified direct method tracking:
    // This is a very basic photometric alignment CONCEPT.
    // A real direct method (like LSD-SLAM or DSO) would involve:
    // - Image pyramids for robust tracking over large motions.
    // - Iterative Gauss-Newton or Levenberg-Marquardt optimization to minimize photometric error.
    // - Depth map estimation (sparse or semi-dense).
    // - Robust outlier rejection.
    // - More complex photometric error calculation using image intensities and gradients.
    Eigen::Matrix4d track(const Frame& new_frame) {
        if (!is_initialized) {
            // First frame, initialize
            last_frame = new_frame; // Copy the frame data
            map->addKeyFrame(new_frame); // Add first frame as keyframe
            current_pose = new_frame.Twc;
            is_initialized = true;
            return current_pose;
        }

        // --- Core Tracking Logic (Highly Simplified Direct Method Placeholder) ---
        // In a real direct method, this would involve:
        // 1. Projecting 3D points from `last_frame` (or map) into `new_frame`.
        // 2. Sampling intensity values from `new_frame` at projected locations.
        // 3. Comparing sampled intensities with `last_frame` intensities.
        // 4. Using image gradients (`last_frame.grad_x`, `last_frame.grad_y`) to compute Jacobian.
        // 5. Iteratively optimizing the 6-DOF camera pose using Gauss-Newton to minimize photometric error.

        // For demonstration, let's assume a small, random motion for now.
        // This simulates some pose change without actual image processing for motion estimation.
        Eigen::Matrix4d relative_transform = Eigen::Matrix4d::Identity();
        
        // Random translation
        relative_transform(0, 3) = dis_trans(gen); // dx
        relative_transform(1, 3) = dis_trans(gen); // dy
        relative_transform(2, 3) = dis_trans(gen); // dz
        
        // Random rotation around Z axis (simplified)
        double angle = dis_rot(gen);
        Eigen::Matrix3d Rz;
        Rz << cos(angle), -sin(angle), 0,
              sin(angle),  cos(angle), 0,
              0,           0,          1;
        relative_transform.block<3,3>(0,0) = Rz;

        // Update global pose: World_T_Current = World_T_Last * Last_T_Current
        current_pose = current_pose * relative_transform; 

        // Update last frame with the new frame's data and its estimated pose
        last_frame = new_frame; 
        last_frame.Twc = current_pose; 

        // Keyframe selection (simplified: add every N frames or based on motion/quality)
        // This is a very basic time-based keyframe selection.
        // A real system would use more sophisticated criteria (e.g., sufficient rotation/translation,
        // number of tracked points, tracking quality degradation).
        if (map->keyframes.empty() || (new_frame.timestamp - map->keyframes.back().timestamp) > 0.5) { // Add keyframe every 0.5 seconds
            map->addKeyFrame(new_frame);
        }

        return current_pose;
    }
};

// --- 7. Viewer Class (Conceptual) ---
// Handles real-time 3D visualization.
// For non-CUDA, this would primarily use OpenGL directly or a lightweight wrapper like Pangolin.
class Viewer {
public:
    Map* map;
    Eigen::Matrix4d current_camera_pose;
    cv::Mat current_image;
    bool running = true;

    Viewer(Map* m) : map(m) {
        // Initialize current_camera_pose to identity
        current_camera_pose = Eigen::Matrix4d::Identity();
    }

    void run() {
        // --- Pangolin Initialization (Uncomment and configure if using Pangolin) ---
        /*
        pangolin::CreateWindowAndBind("Monocular SLAM Viewer", 1024, 768);
        glEnable(GL_DEPTH_TEST);

        // Define Projection and ModelView matrices
        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 420, 420, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, pangolin::AxisY)
        );

        // Choose a display for the 3D view
        pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
        */

        while (running) {
            // --- Pangolin Rendering Loop (Uncomment if using Pangolin) ---
            /*
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);

            // Draw Camera Trajectory
            // Draw current camera pose
            pangolin::glPushMatrix();
            pangolin::OpenGlMatrix M(current_camera_pose);
            pangolin::glMultMatrixd(M.m);
            pangolin::glDrawAxis(0.5); // Draw camera coordinate system
            pangolin::glPopMatrix();

            // Draw keyframe poses
            glColor3f(0.0f, 1.0f, 0.0f); // Green for keyframes
            glLineWidth(2);
            std::lock_guard<std::mutex> lock(g_data_mutex); // Lock map for access
            for (const auto& kf : map->keyframes) {
                pangolin::glPushMatrix();
                pangolin::OpenGlMatrix kf_M(kf.Twc);
                pangolin::glMultMatrixd(kf_M.m);
                pangolin::glDrawAxis(0.2);
                pangolin::glPopMatrix();
            }

            // Draw Map Points (conceptual, as actual 3D points aren't fully implemented yet)
            glPointSize(2);
            glColor3f(1.0f, 0.0f, 0.0f); // Red for map points
            glBegin(GL_POINTS);
            for (const auto& mp : map->map_points) {
                glVertex3d(mp.position.x(), mp.position.y(), mp.position.z());
            }
            glEnd();
            */

            // --- Display Current Camera Image (using OpenCV for 2D overlay) ---
            // This part runs regardless of Pangolin
            {
                std::lock_guard<std::mutex> lock(g_data_mutex); // Lock to access current_image
                if (!current_image.empty()) {
                    cv::imshow("Camera View (Press 'q' to quit)", current_image);
                }
            }
            
            // WaitKey for OpenCV window events and to allow 'q' to quit
            if (cv::waitKey(1) == 'q') {
                running = false;
            }

            // --- Pangolin Finish Frame (Uncomment if using Pangolin) ---
            // pangolin::FinishFrame();

            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Control viewer update rate
        }
        cv::destroyAllWindows(); // Close OpenCV window on exit
    }

    void update(const Eigen::Matrix4d& pose, const cv::Mat& image) {
        std::lock_guard<std::mutex> lock(g_data_mutex);
        current_camera_pose = pose;
        current_image = image.clone(); // Clone to avoid data races if image is modified elsewhere
    }

    void requestStop() {
        running = false;
    }
};

// --- Main SLAM Pipeline ---
int main() {
    // --- System Initialization ---
    Camera camera(0); // Use camera 0 (or specify a video file path: e.g., "video.mp4")
    if (!camera.cap.isOpened()) {
        std::cerr << "Error: Failed to open camera or video file." << std::endl;
        return -1;
    }

    Map map;
    VisualOdometry vo(&map, &camera);
    Viewer viewer(&map);

    // Start viewer in a separate thread
    std::thread viewer_thread(&Viewer::run, &viewer);

    cv::Mat frame_raw;
    double timestamp;
    unsigned long frame_id = 0;

    std::cout << "Starting SLAM pipeline. Press 'q' in 'Camera View' window to quit." << std::endl;

    // --- Main Loop ---
    while (viewer.running) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. Grab Frame
        if (!camera.grabFrame(frame_raw)) {
            std::cerr << "Failed to grab frame. Exiting." << std::endl;
            viewer.requestStop(); // Signal viewer to stop
            break;
        }
        timestamp = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();

        // 2. Undistort Image
        // For simplicity, undistort directly into frame_raw if distCoeffs are non-zero.
        camera.undistort(frame_raw); 
        // If you need a separate undistorted copy: cv::Mat frame_undistorted = frame_raw.clone();

        // 3. Create Frame Object
        Frame current_frame(frame_raw, timestamp); // Pass the (potentially) undistorted frame

        // 4. Track Camera Pose (Visual Odometry)
        // Lock shared data (map) if VO operations directly modify it.
        // In this simple example, VO mostly *reads* from map and *updates* its own pose.
        // Map updates (like addKeyFrame) are protected by a lock inside Map methods.
        Eigen::Matrix4d estimated_pose;
        {
            std::lock_guard<std::mutex> lock(g_data_mutex); // Protect data accessed by both VO and Viewer
            estimated_pose = vo.track(current_frame);
        }

        // 5. Update Viewer
        viewer.update(estimated_pose, current_frame.image);

        // 6. (Optional) Local Mapping & Loop Closure (in separate threads or less frequently)
        // For real-time on CPU, these would be run asynchronously and less often.
        // Example: Run map optimization every 5 seconds in a separate thread
        /*
        static auto last_opt_time = std::chrono::high_resolution_clock::now();
        auto current_loop_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(current_loop_time - last_opt_time).count() >= 5) {
            std::thread([&]() {
                std::lock_guard<std::mutex> lock(g_data_mutex); // Lock map for optimization
                map.optimizeMap();
            }).detach(); // Detach to run in background, care needed for shared data lifetimes
            last_opt_time = current_loop_time;
        }
        */

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // std::cout << "Frame " << frame_id << " processed in " << duration.count() << " ms. FPS: " << 1000.0 / duration.count() << std::endl;
        frame_id++;
    }

    // --- Cleanup ---
    viewer.requestStop(); // Ensure viewer thread is signaled to stop
    if (viewer_thread.joinable()) {
        viewer_thread.join(); // Wait for viewer thread to finish
    }
    std::cout << "SLAM pipeline stopped." << std::endl;

    return 0;
}

/*
--- Compilation Instructions (Linux/macOS) ---

1.  **Install Dependencies:**
    * **OpenCV:**
        ```bash
        sudo apt update
        sudo apt install libopencv-dev
        ```
    * **Eigen:**
        ```bash
        sudo apt install libeigen3-dev
        ```
    * **Pangolin (Optional, for full 3D visualization):**
        If you want the 3D visualization, you'll need Pangolin. Follow its installation guide:
        ```bash
        sudo apt install libglew-dev libglfw3-dev
        git clone [https://github.com/stevenlovegrove/Pangolin.git](https://github.com/stevenlovegrove/Pangolin.git)
        cd Pangolin
        mkdir build && cd build
        cmake ..
        make -j$(nproc)
        sudo make install
        ```

2.  **Compile the code:**

    **Without Pangolin (basic OpenCV `imshow` for 2D camera view):**
    ```bash
    g++ -std=c++17 main.cpp -o monocular_slam \
        `pkg-config --cflags --libs opencv4` \
        -I/usr/include/eigen3 -lpthread -fopenmp
    ```
    * `pkg-config --cflags --libs opencv4` automatically includes OpenCV's compiler flags and libraries. If `opencv4` doesn't work, try `opencv`.
    * `-I/usr/include/eigen3` adds the Eigen library include path.
    * `-lpthread` links the POSIX threads library for multi-threading.
    * `-fopenmp` enables OpenMP for potential parallelization (e.g., in image processing loops).

    **With Pangolin (if installed and uncommented in the code):**
    ```bash
    g++ -std=c++17 main.cpp -o monocular_slam \
        `pkg-config --cflags --libs opencv4` \
        -I/usr/include/eigen3 -lpthread -fopenmp \
        -lPangolin -lGL -lGLEW -lglfw
    ```

--- Key Optimization Points for CPU (as discussed in research) ---

* **OpenMP Parallelization:** Look for computationally intensive loops (e.g., in image gradient computation, feature detection, photometric error calculation in a full direct method). Add `#pragma omp parallel for` to distribute the work across CPU cores.
* **SIMD (Vectorization):** While complex to implement manually, use libraries like OpenCV and Eigen which are often internally optimized with SIMD instructions (SSE/AVX). For custom math, consider Eigen's vectorized operations (`.array()`, `.cwise*()` functions).
* **Efficient Data Structures:** Use `std::vector` for dynamic arrays. Avoid frequent memory reallocations where possible by pre-reserving capacity.
* **Sparse Methods:** The provided `VisualOdometry` is a conceptual direct method. Real direct methods (like LSD-SLAM, DSO) are inherently "sparse" or "semi-dense" in their pixel usage for tracking, focusing on high-gradient areas, which reduces computation compared to dense methods.
* **Reduced Optimization Frequency:** Expensive optimization tasks (e.g., Bundle Adjustment, global pose graph optimization) should be run asynchronously in separate threads and less frequently than the main tracking loop. The example shows a commented-out section for this.
* **Lightweight Visualization:** `cv::imshow` is very lightweight. If using a 3D viewer like Pangolin, ensure rendering is efficient: draw only necessary elements (keyframes, a subset of map points), avoid complex shaders or high-polygon models.
* **Multi-threading:** Decouple the main tracking pipeline from visualization, mapping, and loop closure tasks using `std::thread` and mutexes for shared data.

--- Further Development ---

This code provides a foundational structure. To evolve it into a robust, functional SLAM system, you would need to:

1.  **Implement a Robust Direct Method:** Replace the simplified `VisualOdometry::track` with a proper, iterative photometric alignment algorithm (e.g., a multi-level Gauss-Newton solver minimizing photometric error). This involves:
    * Creating image pyramids for robust tracking over larger motions.
    * Accurate projection and back-projection of points.
    * Depth map estimation (e.g., using inverse depth parametrization).
    * Outlier rejection and robust cost functions.
2.  **Keyframe Management:** Implement more intelligent keyframe selection criteria (e.g., based on sufficient camera motion, number of tracked features/pixels, or tracking quality degradation).
3.  **Local Mapping:** Develop a local mapping module that runs in parallel to triangulate new 3D map points from new keyframes and existing map points, and perform local Bundle Adjustment on a sliding window of keyframes and their observed map points.
4.  **Loop Closure:** Implement a robust loop closure detection (e.g., using a Bag-of-Words approach like DBoW2/3 with ORB features, or a lightweight image descriptor matching) and a global pose graph optimization to correct accumulated drift.
5.  **Camera Calibration:** Load `K` (intrinsic matrix) and `distCoeffs` (distortion coefficients) from a YAML or XML file generated by a precise camera calibration process (e.g., using OpenCV's calibration tools with a chessboard pattern).
6.  **Error Handling and Robustness:** Add more comprehensive error checking, especially for tracking failures, and recovery mechanisms (e.g., re-localization).
7.  **Configuration:** Use external configuration files (e.g., YAML) for system parameters (camera intrinsics, thresholds, map saving paths, etc.).