import pyvista as pv
import numpy as np
'''
X: particle coordinates
point_data: dict of {array name: per-particle array}, e.g. {"w": w, "zeta": zeta}
savepath: path to save the vtp file
'''
def writeVTP(X,point_data,savepath):
    # save particles as a .vtp (PolyData of vertices) with one array per dict entry
    mesh = pv.PolyData(np.asarray(X,dtype=np.float32))
    for name,values in point_data.items():
        mesh.point_data[name] = np.asarray(values)
    mesh.save(savepath)
