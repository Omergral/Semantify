
__all__ = ["myMesh"]
from plyfile import PlyData, PlyElement
import numpy as np

class myMesh(object):
    """
    Attributes:
        v: Vx3 array of vertices
        f: Fx3 array of faces
    """
    def __init__(self, v=None, f=None, filename=None):
        self.v = None
        self.f = None

        if v is not None:
            self.v = np.array(v, dtype=np.float64)
        if f is not None:
            self.f = np.require(f, dtype=np.uint32)

        if filename is not None:
            self.load_from_ply(filename)

        self.vn = None
        self.fn = None
        self.vf = None
        self.v_indexed_by_faces = None
        self.vc = np.array([1.0, 0.0, 0.0])

    def load_from_ply(self, filename):

        plydata = PlyData.read(filename)
        self.plydata = plydata

        self.f = np.vstack(plydata['face'].data['vertex_indices'])
        x = plydata['vertex'].data['x']
        y = plydata['vertex'].data['y']
        z = plydata['vertex'].data['z']
        self.v = np.zeros([x.size, 3])
        self.v[:,0] = x
        self.v[:,1] = y
        self.v[:,2] = z


    def save_ply(self, filename):

        vertex = np.array([tuple(i) for i in self.v], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        face = np.array([(tuple(i), 255, 255, 255) for i in self.f] , 
            dtype=[('vertex_indices', 'i4', (3,)),
            ('red', 'u1'), ('green', 'u1'),
            ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        el2 = PlyElement.describe(face, 'face')
        plydata = PlyData([el, el2])
        plydata.write(filename)

