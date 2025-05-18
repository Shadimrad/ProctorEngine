"""Tiny 2‑D constant‑velocity Kalman smoother."""
import numpy as np
from filterpy.kalman import KalmanFilter

def make_smoother(dt=1/30):
    k=KalmanFilter(dim_x=4,dim_z=2)
    F=np.eye(4);F[0,2]=F[1,3]=dt;k.F,F
    k.H=np.array([[1,0,0,0],[0,1,0,0]]);k.P*=300;k.R*=4
    q=0.3;k.Q=np.array([[q*dt**4/4,0,q*dt**3/2,0],[0,q*dt**4/4,0,q*dt**3/2],[q*dt**3/2,0,q*dt**2,0],[0,q*dt**3/2,0,q*dt**2]],dtype=np.float32)
    def smooth(x,y):
        k.predict();k.update(np.array([x,y]));return float(k.x[0]),float(k.x[1])
    return smooth