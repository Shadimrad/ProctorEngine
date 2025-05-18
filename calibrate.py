"""Robust 5-point calibration routine with better error handling."""
import cv2
import mediapipe as mp
import numpy as np
import time
from head_pose import solve_headpose, cam2world, pixel
from screen_plane import ScreenPlane

# Constants
IRIS_L = 468
EYE_L_CORNER = 33
points_norm = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.5, 0.5)]

def run_calibration(cap, screen_w_mm, screen_h_mm):
    """
    Run a robust 5-point calibration routine with better error handling
    and visual feedback.
    
    Args:
        cap: OpenCV video capture object
        screen_w_mm: Screen width in millimeters
        screen_h_mm: Screen height in millimeters
        
    Returns:
        ScreenPlane object if calibration is successful, None otherwise
    """
    rays = []
    pupils = []
    
    # Initialize MediaPipe face mesh
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    # Create window with a meaningful name
    window_name = 'Calibration - Look at dot and press SPACE'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    current_point = 0
    show_error = False
    error_message = ""
    error_start_time = 0
    error_duration = 2.0  # Show error for 2 seconds
    
    print("\n=== CALIBRATION STARTED ===")
    print("Look at each dot and press SPACE when focused on it")
    print(f"Calibrating point 1/{len(points_norm)}")
    
    while current_point < len(points_norm):
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            time.sleep(0.1)
            continue
            
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create a clean copy for drawing
        display = frame.copy()
        
        # Get current calibration point
        u, v = points_norm[current_point]
        
        # Draw the calibration target
        target_x, target_y = int(u * w), int(v * h)
        cv2.circle(display, (target_x, target_y), 10, (0, 255, 0), -1)
        cv2.circle(display, (target_x, target_y), 20, (0, 255, 0), 2)
        
        # Draw instructions
        cv2.putText(display, f"Point {current_point + 1}/{len(points_norm)}: Look at the green dot", 
                   (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Press SPACE when your gaze is fixed on the dot", 
                   (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Press ESC to cancel calibration", 
                   (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show error message if needed
        if show_error and time.time() - error_start_time < error_duration:
            cv2.putText(display, error_message, (30, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            show_error = False
        
        # Display the frame
        cv2.imshow(window_name, display)
        
        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("Calibration canceled")
            cv2.destroyAllWindows()
            return None
            
        if key == 32:  # SPACE
            # Process the frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                error_message = "No face detected. Please look at the camera and try again."
                show_error = True
                error_start_time = time.time()
                print(error_message)
                continue
            
            # Get face landmarks
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Solve head pose
            ok_hp, R, t = solve_headpose(face_landmarks, w, h)
            
            if not ok_hp:
                error_message = "Head pose estimation failed. Please adjust your position."
                show_error = True
                error_start_time = time.time()
                print(error_message)
                continue
            
            try:
                # Get iris and eye corner positions
                iris_px = np.array(list(pixel(face_landmarks, IRIS_L, w, h)) + [1.0])
                eye_px = np.array(list(pixel(face_landmarks, EYE_L_CORNER, w, h)) + [1.0])
                
                # Camera intrinsic matrix
                cam = np.array(
                    [[w, 0, w / 2],
                     [0, h, h / 2],
                     [0, 0, 1]],
                    dtype=np.float32,
                )
                
                # Convert to camera coordinates
                iris_cam = np.linalg.inv(cam) @ iris_px
                eye_cam = np.linalg.inv(cam) @ eye_px
                
                # Debug prints
                print(f"iris_px: {iris_px}")
                print(f"eye_px: {eye_px}")
                print(f"iris_cam: {iris_cam}")
                print(f"eye_cam: {eye_cam}")
                
                # Normalize coordinates
                if abs(iris_cam[2]) < 1e-9 or abs(eye_cam[2]) < 1e-9:
                    raise ValueError("Division by near-zero in normalization")
                    
                iris_cam = iris_cam / iris_cam[2]
                eye_cam = eye_cam / eye_cam[2]
                
                # Convert to world coordinates
                iris_w = cam2world(iris_cam, R, t)
                eye_w = cam2world(eye_cam, R, t)
                
                # Compute gaze ray
                ray = iris_w - eye_w
                ray_norm = np.linalg.norm(ray)
                
                if ray_norm < 1e-9:
                    raise ValueError("Ray has near-zero length")
                
                # Normalize ray
                ray = ray / ray_norm
                
                # Add to calibration data
                rays.append(ray)
                pupils.append(eye_w)
                
                # Success!
                current_point += 1
                print(f"Point {current_point}/{len(points_norm)} calibrated successfully")
                
                if current_point < len(points_norm):
                    print(f"Now calibrating point {current_point+1}/{len(points_norm)}")
                
                # Show success feedback
                success_frame = display.copy()
                cv2.putText(success_frame, "Point captured successfully!", (30, h - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, success_frame)
                cv2.waitKey(500)  # Display success message for 500ms
                
            except Exception as e:
                error_message = f"Calibration error: {str(e)}"
                show_error = True
                error_start_time = time.time()
                print(f"Error: {e}")
                continue
    
    # Calibration complete
    cv2.destroyAllWindows()
    
    try:
        # Create and fit screen plane
        print("Creating screen plane...")
        plane = ScreenPlane(screen_w_mm, screen_h_mm)
        plane.fit(rays, pupils)
        print("Calibration successful!")
        return plane
    except Exception as e:
        print(f"Error creating screen plane: {e}")
        return None

def main():
    """Simple test function."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
        
    screen_w_mm = 340  # Example values, adjust as needed
    screen_h_mm = 190
    
    plane = run_calibration(cap, screen_w_mm, screen_h_mm)
    
    if plane is not None:
        print("Calibration completed successfully!")
        # Now you could save the plane or use it
    else:
        print("Calibration failed or was canceled.")
        
    cap.release()

if __name__ == "__main__":
    main()