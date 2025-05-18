"""Entry‑point CLI to test each module.

Usage examples:

1. Calibrate plane and save JSON
   python main.py calibrate --out plane.json --screen-w 344 --screen-h 194

2. Run live tracker with saved plane
   python main.py track --plane plane.json
"""
import argparse, json, cv2, mediapipe as mp, numpy as np
from head_pose import solve_headpose, cam2world, pixel
from screen_plane import ScreenPlane
from smoother import make_smoother
from calibrate import run_calibration
from advanced_gaze import AdvancedGaze

IRIS_L=468; EYE_L_CORNER=33

mp_face=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True,max_num_faces=1)

def live_loop_basic(cap, plane):
    """Track gaze using MediaPipe + head pose estimation."""
    smooth = make_smoother()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            okhp, R, t = solve_headpose(lm, w, h)
            if okhp:
                iris_px = np.array(list(pixel(lm, IRIS_L, w, h)) + [0])
                eye_px = np.array(list(pixel(lm, EYE_L_CORNER, w, h)) + [0])
                cam = np.array(
                    [[w, 0, w / 2],
                     [0, h, h / 2],
                     [0, 0, 1]],
                    dtype=np.float32,
                )
                iris_cam = np.linalg.inv(cam) @ iris_px
                eye_cam = np.linalg.inv(cam) @ eye_px
                iris_cam /= iris_cam[2]
                eye_cam /= eye_cam[2]
                iris_w = cam2world(iris_cam, R, t)
                eye_w = cam2world(eye_cam, R, t)
                ray = iris_w - eye_w
                ray /= np.linalg.norm(ray)
                hit = plane.intersect(eye_w, ray)
                if hit:
                    u, v, _ = hit
                    px, py = int(u * w), int(v * h)
                    smx, smy = smooth(px, py)
                    color = (0, 255, 0) if 0 <= u <= 1 and 0 <= v <= 1 else (0, 0, 255)
                    cv2.circle(frame, (int(smx), int(smy)), 8, color, -1)
        cv2.imshow('gaze', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def live_loop_advanced(cap):
    """Track gaze using the AdvancedGaze wrapper."""
    smooth = make_smoother()
    gaze = AdvancedGaze()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = gaze.process(frame)
        if res:
            u, v = res
            px, py = int(u * w), int(v * h)
            smx, smy = smooth(px, py)
            cv2.circle(frame, (int(smx), int(smy)), 8, (0, 255, 0), -1)
        cv2.imshow('gaze', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    cal=sub.add_parser('calibrate');
    cal.add_argument('--out', required=True)
    cal.add_argument('--screen-w', type=float, required=True)
    cal.add_argument('--screen-h', type=float, required=True)
    cal.add_argument('--cam', type=int, default=0)

    tr=sub.add_parser('track');
    tr.add_argument('--plane')
    tr.add_argument('--cam', type=int, default=0)
    tr.add_argument('--method', choices=['basic', 'advanced'], default='basic')
    args=ap.parse_args()
    cap=cv2.VideoCapture(args.cam)
    if not cap.isOpened(): raise SystemExit('Camera failed')
    if args.cmd=='calibrate':
        plane=run_calibration(cap,args.screen_w,args.screen_h)
        with open(args.out,'w') as f: json.dump({'O':plane.O.tolist(),'Ux':plane.Ux.tolist(),'Uy':plane.Uy.tolist(),'n':plane.n.tolist(),'w':args.screen_w,'h':args.screen_h},f)
        print('✓ saved ',args.out)
    else:
        if args.method == 'basic':
            if not args.plane:
                raise SystemExit('--plane is required for basic tracking')
            data = json.load(open(args.plane))
            plane = ScreenPlane(data['w'], data['h'])
            plane.O = np.array(data['O'])
            plane.Ux = np.array(data['Ux'])
            plane.Uy = np.array(data['Uy'])
            plane.n = np.array(data['n'])
            live_loop_basic(cap, plane)
        else:
            live_loop_advanced(cap)

if __name__=='__main__':
    main()
