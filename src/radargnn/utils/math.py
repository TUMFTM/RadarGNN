import numpy as np
from scipy.spatial import ConvexHull
from math import sqrt
from math import atan2, cos, sin, pi
from collections import namedtuple
import torch


def get_box_corners(x, y, l, w, theta) -> np.array:
    """ Returns the corner points of a rotated bounding box.

    Args:
        x: Position of the box center in x dimension.
        y: Position of the box center in y dimension.
        l: Length of the box.
        w: Width of the box.
        theta: Rotation (yaw) of the box in degrees.

    Returns:
        corners: Corner points of the box (x, y).
    """
    # corners relative to zero origin - long side along x axis
    c1_orig = np.array([l / 2, w / 2]).reshape(1, 2)
    c2_orig = np.array([l / 2, -w / 2]).reshape(1, 2)
    c3_orig = np.array([-l / 2, -w / 2]).reshape(1, 2)
    c4_orig = np.array([-l / 2, w / 2]).reshape(1, 2)
    corners_orig = np.concatenate((c1_orig, c2_orig, c3_orig, c4_orig), axis=0)

    # rotate bounding box corners by theta
    theta_rad = (theta * np.pi) / 180
    rot = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                    [np.sin(theta_rad), np.cos(theta_rad)]])

    corners_orig = np.matmul(rot, corners_orig.T).T

    # absolute center of bounding box using point coordinates
    c_abs = np.array([x, y]).reshape(1, 2)

    # translate (rotated) corners to get absolute position of bb corners
    c1_bb = corners_orig[0, :].reshape(1, 2) + c_abs
    c2_bb = corners_orig[1, :].reshape(1, 2) + c_abs
    c3_bb = corners_orig[2, :].reshape(1, 2) + c_abs
    c4_bb = corners_orig[3, :].reshape(1, 2) + c_abs
    corners = np.concatenate((c1_bb, c2_bb, c3_bb, c4_bb), axis=0)

    return corners


def box_area_rotated(box_matrix_tensor: torch.Tensor) -> torch.Tensor:
    """ Returns the area of all boxes in the matrix

    Args:
        box_matrix_tensor: needs to be a matrix containing one absolute rotated bounding box per row.
            The absolute rotated bounding box needs to be of format: [x, y, l, w, theta].

    """

    return box_matrix_tensor[:, 2] * box_matrix_tensor[:, 3]


def is_point_in_rect(rect: np.array, point: np.array) -> bool:
    """ Determines whether a point is within a rectangle or not.

    Note:
        Corners of rect must be defined clockwise or counterclockwise

    Args:
        rect: Rectangle defined by four corner points.
        point: Coordinates of the point to test.

    Returns:
        True if point in rect, else false.
    """

    xP = point[0]
    yP = point[1]

    xA = rect[0, 0]
    yA = rect[0, 1]
    xB = rect[1, 0]
    yB = rect[1, 1]
    xC = rect[2, 0]
    yC = rect[2, 1]
    xD = rect[3, 0]
    yD = rect[3, 1]

    abcd = 0.5 * abs((yA - yC) * (xD - xB) + (yB - yD) * (xA - xC))

    abp = 0.5 * abs(xA * (yB - yP) + xB * (yP - yA) + xP * (yA - yB))
    bcp = 0.5 * abs(xB * (yC - yP) + xC * (yP - yB) + xP * (yB - yC))
    cdp = 0.5 * abs(xC * (yD - yP) + xD * (yP - yC) + xP * (yC - yD))
    dap = 0.5 * abs(xD * (yA - yP) + xA * (yP - yD) + xP * (yD - yA))

    sum_tri = abp + bcp + cdp + dap

    return ((sum_tri - abcd) < 1E-6)


def get_points_in_rotated_box(box: torch.Tensor, points: torch.Tensor) -> np.ndarray:
    """ Determines for multiple points, whether they are within a rotated rectangle or not.

    Args:
        box: Bounding box in the form of [x, y, l, w, theta_x(degree)].
        points: Tensor with the coordinates of multiple points.

    Returns:
        Tensor of all points within the box.
    """

    # transform to numpy array for maximum floating point precision in conversion to absolute box
    if isinstance(box, torch.Tensor):
        box = box.detach().numpy()

    # transform box to get the four corners coordinates
    corners = get_box_corners(box[0], box[1], box[2], box[3], box[4])

    # get all points within the box
    points_in_rect_idx = []
    for i, point in enumerate(points):
        if is_point_in_rect(corners, point):
            points_in_rect_idx.append(i)
        else:
            continue

    return points[points_in_rect_idx, :]


def get_points_in_box(box: torch.Tensor, points: np.ndarray) -> np.ndarray:
    """ Determines for multiple points, whether they are within a aligned rectangle or not.

    Args:
        box: Bounding box in the form of [x_min, y_min, x_max, y_max].
        points: Array with the coordinates of multiple points.

    Returns:
        Array of all points within the box.
    """
    x_min = box[0].item()
    y_min = box[1].item()
    x_max = box[2].item()
    y_max = box[3].item()

    box_idx = np.where((points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (
        points[:, 1] >= y_min) & (points[:, 1] <= y_max))[0]

    box_points = points[box_idx, :]
    return box_points


def get_stats_of_predicted_box_points(box_points_predict: np.ndarray, box_points_true: np.ndarray) -> tuple:
    """ Calculates statistics for two sets of points.

    Given two sets of points (correct and wrong predictions):
        - returns tp, fp, and fn predictions
    """
    aset = set([tuple(x) for x in box_points_predict])
    bset = set([tuple(x) for x in box_points_true])
    intersection = np.array([x for x in aset & bset])

    tp = intersection.shape[0]
    fn = box_points_true.shape[0] - intersection.shape[0]
    fp = box_points_predict.shape[0] - intersection.shape[0]

    return tp, fp, fn


def get_discrete_iou(tp, fp, fn):
    """ Calculates the discrete point based IOU.
    """
    if (tp + fp + fn) != 0:
        return tp / (tp + fp + fn)
    else:
        return 0.00001


def point_iou(boxes_pred: torch.Tensor, boxes_gt: torch.Tensor, points: np.ndarray, box_aligned: bool) -> torch.Tensor:
    """ Calculates a discrete / point-based IOU of two bounding boxes.

    Args:
        boxes_pred: Matrix containing one predicted bounding box per row.
        boxes_gt: Matrix containing one ground truth bounding box per row.
        points: Coordinates of all points in the graph.
        box_aligned: Bool whether box is aligned or rotated.

    Returns:
        iou_matrix: Matrix of point-iou's of the boxes
    """

    iou_matrix = np.empty([boxes_pred.shape[0], boxes_gt.shape[0]])

    for i in range(boxes_pred.shape[0]):
        box_pred = boxes_pred[i, :]
        for j in range(boxes_gt.shape[0]):
            box_true = boxes_gt[j, :]

            if box_aligned:
                # points for box if box is aligned
                box_points_predict = get_points_in_box(box_pred, points)
                box_points_true = get_points_in_box(box_true, points)
            else:
                # points in box if box is rotated
                box_points_predict = get_points_in_rotated_box(box_pred, points)
                box_points_true = get_points_in_rotated_box(box_true, points)

            tp, fp, fn = get_stats_of_predicted_box_points(
                box_points_predict, box_points_true)
            iou = get_discrete_iou(tp, fp, fn)
            iou_matrix[i, j] = iou

    iou_matrix = torch.from_numpy(iou_matrix)
    return iou_matrix


def minimum_bounding_rectangle_with_rotation(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates

    ATTENTION: THIS FUNCTION IS NOT 100% CORRECT
    -> ONLY USE THE FUNCTION BELOW: "minimum_bounding_rectangle_with_rotation_alternative()"
    """

    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def minimum_bounding_rectangle_without_rotation(points):
    """ Returns the smallest enclosing aligned rectangle for the given points.
    """

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    c1 = np.array([x_min, y_min]).reshape(1, 2)
    c2 = np.array([x_min, y_max]).reshape(1, 2)
    c3 = np.array([x_max, y_min]).reshape(1, 2)
    c4 = np.array([x_max, y_max]).reshape(1, 2)

    rval = np.concatenate((c1, c2, c3, c4), axis=0)
    return rval


# %% ALL CODE BELOW IS FROM: https://bitbucket.org/william_rusnack/minimumboundingbox/src/master/

def minimum_bounding_rectangle_with_rotation_alternative(points):
    bounding_box = MinimumBoundingBox(points)  # returns namedtuple
    bounding_box.corner_points

    rval = None
    for corner in bounding_box.corner_points:
        if rval is None:
            rval = np.array([corner])
        else:
            rval = np.concatenate([rval, np.array([corner])], axis=0)

    return rval


def unit_vector(pt0, pt1):
    # returns an unit vector that points in the direction of pt0 to pt1
    dis_0_to_1 = sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, \
           (pt1[1] - pt0[1]) / dis_0_to_1


def orthogonal_vector(vector):
    # from vector returns a orthogonal/perpendicular vector of equal length
    return -1 * vector[1], vector[0]


def bounding_area(index, hull):
    unit_vector_p = unit_vector(hull[index], hull[index + 1])
    unit_vector_o = orthogonal_vector(unit_vector_p)

    dis_p = tuple(np.dot(unit_vector_p, pt) for pt in hull)
    dis_o = tuple(np.dot(unit_vector_o, pt) for pt in hull)

    min_p = min(dis_p)
    min_o = min(dis_o)
    len_p = max(dis_p) - min_p
    len_o = max(dis_o) - min_o

    return {'area': len_p * len_o,
            'length_parallel': len_p,
            'length_orthogonal': len_o,
            'rectangle_center': (min_p + len_p / 2, min_o + len_o / 2),
            'unit_vector': unit_vector_p,
            }


def to_xy_coordinates(unit_vector_angle, point):
    # returns converted unit vector coordinates in x, y coordinates
    angle_orthogonal = unit_vector_angle + pi / 2
    return point[0] * cos(unit_vector_angle) + point[1] * cos(angle_orthogonal), point[0] * sin(unit_vector_angle) + point[1] * sin(angle_orthogonal)


def rotate_points(center_of_rotation, angle, points):
    # Requires: center_of_rotation to be a 2d vector. ex: (1.56, -23.4)
    #           angle to be in radians
    #           points to be a list or tuple of points. ex: ((1.56, -23.4), (1.56, -23.4))
    # Effects: rotates a point cloud around the center_of_rotation point by angle
    rot_points = []
    ang = []
    for pt in points:
        diff = tuple([pt[d] - center_of_rotation[d] for d in range(2)])
        diff_angle = atan2(diff[1], diff[0]) + angle
        ang.append(diff_angle)
        diff_length = sqrt(sum([d**2 for d in diff]))
        rot_points.append((center_of_rotation[0] + diff_length * cos(diff_angle),
                           center_of_rotation[1] + diff_length * sin(diff_angle)))

    return rot_points


def rectangle_corners(rectangle):
    # Requires: the output of mon_bounding_rectangle
    # Effects: returns the corner locations of the bounding rectangle
    corner_points = []
    for i1 in (.5, -.5):
        for i2 in (i1, -1 * i1):
            corner_points.append((rectangle['rectangle_center'][0] + i1 * rectangle['length_parallel'],
                                 rectangle['rectangle_center'][1] + i2 * rectangle['length_orthogonal']))

    return rotate_points(rectangle['rectangle_center'], rectangle['unit_vector_angle'], corner_points)


BoundingBox = namedtuple('BoundingBox', ('area',
                                         'length_parallel',
                                         'length_orthogonal',
                                         'rectangle_center',
                                         'unit_vector',
                                         'unit_vector_angle',
                                         'corner_points'
                                         )
                         )


# use this function to find the listed properties of the minimum bounding box of a point cloud
def MinimumBoundingBox(points):
    # Requires: points to be a list or tuple of 2D points. ex: ((5, 2), (3, 4), (6, 8))
    #           needs to be more than 2 points
    # Effects:  returns a namedtuple that contains:
    #               area: area of the rectangle
    #               length_parallel: length of the side that is parallel to unit_vector
    #               length_orthogonal: length of the side that is orthogonal to unit_vector
    #               rectangle_center: coordinates of the rectangle center
    #                   (use rectangle_corners to get the corner points of the rectangle)
    #               unit_vector: direction of the length_parallel side. RADIANS
    #                   (it's orthogonal vector can be found with the orthogonal_vector function
    #               unit_vector_angle: angle of the unit vector
    #               corner_points: set that contains the corners of the rectangle

    if len(points) <= 2:
        raise ValueError('More than two points required.')

    hull_ordered = [points[index] for index in ConvexHull(points).vertices]
    hull_ordered.append(hull_ordered[0])
    hull_ordered = tuple(hull_ordered)

    min_rectangle = bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered) - 1):
        rectangle = bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    min_rectangle['unit_vector_angle'] = atan2(
        min_rectangle['unit_vector'][1], min_rectangle['unit_vector'][0])
    min_rectangle['rectangle_center'] = to_xy_coordinates(
        min_rectangle['unit_vector_angle'], min_rectangle['rectangle_center'])

    # this is ugly but a quick hack and is being changed in the speedup branch
    return BoundingBox(
        area=min_rectangle['area'],
        length_parallel=min_rectangle['length_parallel'],
        length_orthogonal=min_rectangle['length_orthogonal'],
        rectangle_center=min_rectangle['rectangle_center'],
        unit_vector=min_rectangle['unit_vector'],
        unit_vector_angle=min_rectangle['unit_vector_angle'],
        corner_points=set(rectangle_corners(min_rectangle))
    )
