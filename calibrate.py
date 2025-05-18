"""Interactive 5‑dot calibration routine (CLI)."""
import cv2, mediapipe as mp, numpy as np
from head_pose import solve_headpose, cam2world, pixel
from screen_plane import ScreenPlane

IRIS_L=468; EYE_L_CORNER=33

points_norm=[(0.1,0.1),(0.9,0.1),(0.9,0.9),(0.1,0.9),(0.5,0.5)]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['clicked'] = True

def run_calibration(cap, screen_w_mm, screen_h_mm):
    rays=[]; pupils=[]
    mp_face=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True,max_num_faces=1)
    cv2.namedWindow('calib')
    mouse_state = {'clicked': False}
    cv2.setMouseCallback('calib', mouse_callback, mouse_state)
    
    while len(rays)<5:
        ok,frame=cap.read()
        if not ok: continue
        h,w=frame.shape[:2]
        idx=len(rays); u,v=points_norm[idx]
        cv2.circle(frame,(int(u*w),int(v*h)),12,(0,255,0),-1)
        cv2.putText(frame,'Look at dot & click mouse to calibrate',(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('calib',frame)
        
        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # ESC
            raise KeyboardInterrupt
            
        if mouse_state['clicked']:
            mouse_state['clicked'] = False  # Reset the click state
            res=mp_face.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                print("Face/iris not detected, try again!")
                cv2.putText(frame,'Face/iris not detected, try again!',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv2.imshow('calib',frame)
                cv2.waitKey(500)
                continue
            lm=res.multi_face_landmarks[0].landmark
            okhp,R,t=solve_headpose(lm,w,h)
            if not okhp:
                print("Head pose estimation failed, try again!")
                cv2.putText(frame,'Head pose failed, try again!',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv2.imshow('calib',frame)
                cv2.waitKey(500)
                continue
            iris_px=np.array(list(pixel(lm,IRIS_L,w,h))+[0])
            eye_px=np.array(list(pixel(lm,EYE_L_CORNER,w,h))+[0])
            cam = np.array(
                [[w, 0, w / 2],
                 [0, h, h / 2],
                 [0, 0, 1]],
                dtype=np.float32,
            )
            iris_cam=np.linalg.inv(cam)@iris_px; eye_cam=np.linalg.inv(cam)@eye_px
            if iris_cam[2] != 0 and eye_cam[2] != 0:
                iris_cam/=iris_cam[2]; eye_cam/=eye_cam[2]
                iris_w=cam2world(iris_cam,R,t); eye_w=cam2world(eye_cam,R,t)
                ray=iris_w-eye_w
                ray_norm = np.linalg.norm(ray)
                if ray_norm > 0 and np.all(np.isfinite(ray)) and np.all(np.isfinite(eye_w)):
                    ray/=ray_norm
                    rays.append(ray); pupils.append(eye_w)
                    print(f"Collected {len(rays)} valid rays...")
    
    cv2.destroyWindow('calib')
    plane=ScreenPlane(screen_w_mm,screen_h_mm); plane.fit(rays,pupils)
    return plane