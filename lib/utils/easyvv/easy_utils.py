# easymocap utility functions
import os
import cv2
import numpy as np
from lib.utils.easyvv.base_utils import dotdict


class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out + '\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.10f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == 'real':
            self._write('{}: {:.10f}'.format(key, value))  # as accurate as possible
        else:
            raise NotImplementedError

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'real':
            output = self.fs.getNode(key).real()
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_camera(intri_path: str, extri_path: str, cam_names=[], read_rvec=True):
    assert os.path.exists(intri_path), intri_path
    assert os.path.exists(extri_path), extri_path

    intri = FileStorage(intri_path)
    extri = FileStorage(extri_path)
    cams = dotdict()
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # Intrinsics
        cams[cam] = dotdict()
        cams[cam].K = intri.read('K_{}'.format(cam))
        cams[cam].H = intri.read('H_{}'.format(cam), dt='real') or -1
        cams[cam].W = intri.read('W_{}'.format(cam), dt='real') or -1
        cams[cam].invK = np.linalg.inv(cams[cam]['K'])

        # Extrinsics
        Tvec = extri.read('T_{}'.format(cam))
        Rvec = extri.read('R_{}'.format(cam))
        if Rvec is not None and read_rvec: R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri.read('Rot_{}'.format(cam))
            Rvec = cv2.Rodrigues(R)[0]
        RT = np.hstack((R, Tvec))

        cams[cam].R = R
        cams[cam].T = Tvec
        cams[cam].C = - Rvec.T @ Tvec
        cams[cam].RT = RT
        cams[cam].Rvec = Rvec
        cams[cam].P = cams[cam].K @ cams[cam].RT

        # Distortion
        D = intri.read('D_{}'.format(cam))
        if D is None: D = intri.read('dist_{}'.format(cam))
        cams[cam].D = D

        # Time input
        cams[cam].t = extri.read('t_{}'.format(cam), dt='real') or 0  # temporal index, might all be 0

        # Bounds, could be overwritten
        cams[cam].n = extri.read('n_{}'.format(cam), dt='real') or 0.0001  # temporal index, might all be 0
        cams[cam].f = extri.read('f_{}'.format(cam), dt='real') or 1e6  # temporal index, might all be 0
        cams[cam].bounds = extri.read('bounds_{}'.format(cam))
        cams[cam].bounds = np.array([[-1e6, -1e6, -1e6], [1e6, 1e6, 1e6]]) if cams[cam].bounds is None else cams[cam].bounds

        # CCM
        cams[cam].ccm = intri.read('ccm_{}'.format(cam))
        cams[cam].ccm = np.eye(3) if cams[cam].ccm is None else cams[cam].ccm

    return dotdict(cams)


def write_camera(cameras: dict, path: str, intri_path: str = '', extri_path: str = ''):
    from os.path import join
    os.makedirs(path, exist_ok=True)
    if not intri_path or not extri_path:
        intri_name = join(path, 'intri.yml')  # TODO: make them arguments
        extri_name = join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    cam_names = [key_.split('.')[0] for key_ in cameras.keys()]
    intri.write('names', cam_names, 'list')
    extri.write('names', cam_names, 'list')

    cameras = dotdict(cameras)
    for key_, val in cameras.items():
        if key_ == 'basenames': continue
        key = key_.split('.')[0]
        # Intrinsics
        intri.write('K_{}'.format(key), val.K)
        if 'H' in val: intri.write('H_{}'.format(key), val.H, 'real')
        if 'W' in val: intri.write('W_{}'.format(key), val.W, 'real')

        # Distortion
        if 'D' not in val:
            if 'dist' in val: val.D = val.dist
            else: val.D = np.zeros((5, 1))
        intri.write('D_{}'.format(key), val.D.reshape(5, 1))

        # Extrinsics
        if 'R' not in val: val.R = cv2.Rodrigues(val.Rvec)[0]
        if 'Rvec' not in val: val.Rvec = cv2.Rodrigues(val.R)[0]
        extri.write('R_{}'.format(key), val.Rvec)
        extri.write('Rot_{}'.format(key), val.R)
        extri.write('T_{}'.format(key), val.T.reshape(3, 1))

        # Temporal
        if 't' in val: extri.write('t_{}'.format(key), val.t, 'real')

        # Bounds
        if 'n' in val: extri.write('n_{}'.format(key), val.n, 'real')
        if 'f' in val: extri.write('f_{}'.format(key), val.f, 'real')
        if 'bounds' in val: extri.write('bounds_{}'.format(key), val.bounds)

        # Color correction matrix
        if 'ccm' in val: intri.write('ccm_{}'.format(key), val.ccm)
