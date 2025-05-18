"""Head‑pose and gaze‑ray utilities."""
import numpy as np, cv2, mediapipe as mp

FM = mp.solutions.face_mesh
RIGID_IDX = [1, 33, 263, 61, 291, 199]
MODEL_3D = np.array([[0,0,0],[-30,35,-30],[30,35,-30],[-25,-30,-15],[25,-30,-15],[0,-65,-5]],dtype=np.float32)

def pixel(lmks, idx, w, h):
    l = lmks[idx]; return l.x*w, l.y*h

def solve_headpose(lmks, w, h):
    """Return (ok, R (3×3), t (3,))"""
    img_pts = np.array([pixel(lmks,i,w,h) for i in RIGID_IDX],dtype=np.float32)
    cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float32)
    ok,rvec,tvec=cv2.solvePnP(MODEL_3D,img_pts,cam,None,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return False,None,None
    R,_=cv2.Rodrigues(rvec)
    return True,R,tvec.flatten()

def cam2world(pt_cam,R,t):
    return R.T@(pt_cam - t)