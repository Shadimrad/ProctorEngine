"""Physical screen plane solved from 5 gaze rays."""
import numpy as np

class ScreenPlane:
    def __init__(self, width_mm, height_mm):
        self.w_mm=width_mm; self.h_mm=height_mm
        self.O=self.Ux=self.Uy=self.n=None
    @staticmethod
    def _pca_normal(rays):
        _,_,vt=np.linalg.svd(np.stack(rays));n=vt[-1];return n/np.linalg.norm(n)
    def fit(self,rays,pupils):
        self.n=self._pca_normal(rays)
        t0=-np.dot(pupils[4],self.n)/np.dot(rays[4],self.n)
        centre=pupils[4]+t0*rays[4]
        t1=-np.dot(pupils[1],self.n)/np.dot(rays[1],self.n)
        tr=pupils[1]+t1*rays[1]
        Ux=tr-centre; Ux-=self.n*np.dot(Ux,self.n); Ux/=np.linalg.norm(Ux)
        Uy=np.cross(self.n,Ux); Uy/=np.linalg.norm(Uy)
        self.O=centre-0.5*Ux*self.w_mm-0.5*Uy*self.h_mm
        self.Ux=Ux*self.w_mm; self.Uy=Uy*self.h_mm
    def intersect(self,pupil,ray):
        denom=float(np.dot(ray,self.n))
        if abs(denom)<1e-6: return None
        t=np.dot(self.O-pupil,self.n)/denom
        if t<0: return None
        P=pupil+t*ray; rel=P-self.O
        u=np.dot(rel,self.Ux)/(self.w_mm**2)
        v=np.dot(rel,self.Uy)/(self.h_mm**2)
        return u,v,P