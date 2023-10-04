import numpy as np
import cv2
from sklearn.decomposition import PCA

def vis_point_cloud(points, colors=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None: pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    
def pcd_array2o3d(points, colors=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None: pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def compute_pca_features(
    feature2D_map, feature_rescal_factor=8, save_path=None
):
    """
    feature2D_map: H*W*c
    feature3D: N*C
    """
    # PCA decrease dim of input feature:
    pca = PCA(n_components=3, svd_solver="arpack")

    H, W, C = feature2D_map.shape

    # feat_concated = np.concatenate(
        # [ feature2D_map.reshape(-1, C)], axis=0
    # )  # 2N * C
    pca_feature = pca.fit_transform(feature2D_map.reshape(-1, C))

    # Convert decreased feature to color map:
    pca_feature = cv2.normalize(
        pca_feature,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC3,
    )
    feature2D_map_pca = pca_feature.reshape(H, W, 3)
    return feature2D_map_pca

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def plt_show_bbox(img, bounds, ext, ixt, path):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    bbox_3d = get_bound_corners(bounds)
    bbox_2d = project(bbox_3d, ixt, ext)
    lines = np.array([[0, 0, 0, 7, 7, 7, 1, 1, 2, 2, 4, 4], [1, 2, 4, 5, 6, 3, 3, 5, 3, 6, 5, 6]]).T
    for id, line in enumerate(lines):
        pt1 = bbox_2d[line[0]]
        pt2 = bbox_2d[line[1]]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'm')
    plt.savefig(path)

def bbox_img(image, bounds, ext, ixt, linewidth=3, color=(0, 255, 0)):
    bbox_3d = get_bound_corners(bounds)
    bbox_2d = project(bbox_3d, ixt, ext)
    lines = np.array([[0, 0, 0, 7, 7, 7, 1, 1, 2, 2, 4, 4], [1, 2, 4, 5, 6, 3, 3, 5, 3, 6, 5, 6]]).T
    for id, line in enumerate(lines):
        pt1 = bbox_2d[line[0]].astype(int)
        pt2 = bbox_2d[line[1]].astype(int)
        image = cv2.line(image, tuple(pt1), tuple(pt2), color, linewidth)
    return image
            
def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T

    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

row_col_ = {
    2: (2, 1),
    7: (2, 4),
    8: (2, 4),
    9: (3, 3),
    26: (4, 7)
}

row_col_square = {
    2: (2, 1),
    7: (3, 3),
    8: (3, 3),
    9: (3, 3),
    26: (5, 5)
}

def get_row_col(l, square):
    if square and l in row_col_square.keys():
        return row_col_square[l]
    if l in row_col_.keys():
        return row_col_[l]
    else:
        from math import sqrt
        row = int(sqrt(l) + 0.5)
        col = int(l/ row + 0.5)
        if row*col<l:
            col = col + 1
        if row > col:
            row, col = col, row
        return row, col

def merge(images, row=-1, col=-1, resize=False, ret_range=False, square=False, **kwargs):
    if row == -1 and col == -1:
        row, col = get_row_col(len(images), square)
    height = images[0].shape[0]
    width = images[0].shape[1]
    # special case
    if height > width:
        if len(images) == 3:
            row, col = 1, 3
    if len(images[0].shape) > 2:
        ret_img = np.zeros((height * row, width * col, images[0].shape[2]), dtype=np.uint8) + 255
    else:
        ret_img = np.zeros((height * row, width * col), dtype=np.uint8) + 255
    ranges = []
    for i in range(row):
        for j in range(col):
            if i*col + j >= len(images):
                break
            img = images[i * col + j]
            # resize the image size
            img = cv2.resize(img, (width, height))
            ret_img[height * i: height * (i+1), width * j: width * (j+1)] = img
            ranges.append((width*j, height*i, width*(j+1), height*(i+1)))
    if resize:
        min_height = 1000
        if ret_img.shape[0] > min_height:
            scale = min_height/ret_img.shape[0]
            ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
    if ret_range:
        return ret_img, ranges
    return ret_img
