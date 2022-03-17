import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def obtain_image_hull(image: np.array, points: np.array):
    points = points.squeeze()
    hull = ConvexHull(points)
    # deln = Delaunay(points[hull.vertices])
    # idx = np.stack(np.indices(image.shape), axis=-1)
    # out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    # out_img = np.zeros_like(image)
    # out_img[out_idx] = 1
    return image, hull


def pix2meter(dist):
    # m8 = np.array([[484, 280], [1187, 578], [1873, 871], [2541, 1151], [3199, 1431]])
    # m5 = np.array([[3115,376], [2755, 537], [2407, 697], [2059, 854], [1715, 1007], [1374, 1156]])
    # dist8 = np.array([np.linalg.norm(m8[ind, :] - m8, axis=1) for ind in range(len(m8))])
    # dist5 = np.array([np.linalg.norm(m5[ind, :] - m5, axis=1) for ind in range(len(m5))])
    # dist1 = np.array([])
    # for ind in range(1, len(dist8)):
    #     dist1 = np.hstack((dist1, np.diag(dist8, ind) / (ind * 2)))
    # for ind in range(1, len(dist5)):
    #     dist1 = np.hstack((dist1, np.diag(dist5, ind) / ind))
    # np.mean(dist1)
    conversion_rate = 376.1634939248632 #pxels/meter
    return dist/conversion_rate


def real_distance(coord: np.array) -> np.array:
    """
    Returns the vector distance(s) in the real world

    :param coord: numpy.array with [[x,y], ..., [x_n, y_n]] coordinates
    :return: converted numpy.array [d1, ..., d_n]
    """
    x = coord[:, 0]
    y = coord[:, 1]
    return pix2meter(np.hypot(x, y))