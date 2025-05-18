"""Interactive 5‑dot calibration routine (CLI)."""
import cv2, mediapipe as mp, numpy as np
from head_pose import solve_headpose, cam2world, pixel
from screen_plane import ScreenPlane

IRIS_L=468; EYE_L_CORNER=33

points_norm=[(0.1,0.1),(0.9,0.1),(0.9,0.9),(0.1,0.9),(0.5,0.5)]

def run_calibration(cap, screen_w_mm, screen_h_mm):
    rays=[]; pupils=[]
    mp_face=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True,max_num_faces=1)
    while len(rays)<5:
        ok,frame=cap.read();
        if not ok: continue
        h,w=frame.shape[:2]
        idx=len(rays); u,v=points_norm[idx]
        cv2.circle(frame,(int(u*w),int(v*h)),12,(0,255,0),-1)
        cv2.putText(frame,'Look & press SPACE',(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('calib',frame)
        k=cv2.waitKey(1)&0xFF
        print(f"Key pressed: {k}")  # Debug print
        if k==27: raise KeyboardInterrupt
        if k==255: continue  # No key pressed
        res=mp_face.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks: continue
        lm=res.multi_face_landmarks[0].landmark
        okhp,R,t=solve_headpose(lm,w,h)
        if not okhp: continue
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