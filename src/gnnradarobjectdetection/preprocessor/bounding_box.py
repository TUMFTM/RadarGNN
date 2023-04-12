import numpy as np
from math import atan2
from abc import ABC, abstractmethod

from gnnradarobjectdetection.utils.math import get_box_corners


class RelativeBoundingBox(ABC):

    @abstractmethod
    def get_absolute_bounding_box(self, x: float, y: float):
        """ Transforms the bounding box to an absolute representation defined by its four corners.

        Arguments:
            x: X-position of the radar point to which the bounding box corresponds.
            y: Y-position of the radar point to which the bounding box corresponds.
        """
        return


class AbsoluteRotatedBoundingBox(RelativeBoundingBox):
    """ Absolute rotated bounding box.

    Attributes:
        x: Absolute x-position of the bounding box center.
        y: Absolute y-position of the bounding box center.
        l: Length of the box.
        w: Width of the box.
        theta: Angle between longitudinal box-axis and x-axis of the coordinate system.
    """
    def __init__(self, x, y, l, w, theta):
        self.x = x
        self.y = y
        self.l = l
        self.w = w
        self.theta = theta

    def get_absolute_bounding_box(self):
        # corners relative to zero origin - long side along x axis
        c1_orig = np.array([self.l / 2, self.w / 2]).reshape(1, 2)
        c2_orig = np.array([self.l / 2, - self.w / 2]).reshape(1, 2)
        c3_orig = np.array([- self.l / 2, - self.w / 2]).reshape(1, 2)
        c4_orig = np.array([- self.l / 2, self.w / 2]).reshape(1, 2)
        corners_orig = np.concatenate(
            (c1_orig, c2_orig, c3_orig, c4_orig), axis=0)

        # rotate bounding box corners by theta
        theta_x = self.theta
        theta_rad = (theta_x * np.pi) / 180
        rot = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                        [np.sin(theta_rad), np.cos(theta_rad)]])

        corners_orig = np.matmul(rot, corners_orig.T).T

        # absolute center of bounding box
        c_abs = np.array([self.x, self.y]).reshape(1, 2)

        # translate (rotated) corners to get absolute position of bb corners
        c1_bb = corners_orig[0, :].reshape(1, 2) + c_abs
        c2_bb = corners_orig[1, :].reshape(1, 2) + c_abs
        c3_bb = corners_orig[2, :].reshape(1, 2) + c_abs
        c4_bb = corners_orig[3, :].reshape(1, 2) + c_abs
        corners = np.concatenate((c1_bb, c2_bb, c3_bb, c4_bb), axis=0)

        return BoundingBox(corners, False)


class RotationInvariantRelativeRotatedBoundingBox(RelativeBoundingBox):
    """ Rotation invariant bounding box.

    It is defined relative to the radar point to which it corresponds to.
    Furthermore, it is defined relative to the connection vector of the radar point and its nearest neighbor.

    Attributes:
        d: Distance between the point and the box center.
        theta_v_p_nn_v_p_c: Angle between the vector point->nearest-neighbor and the vector point->bounding-box-center
        l: Length of the box.
        w: Width of the box.
        theta_v_p_nn_v_dir: Angle between the vector point->nearest-neighbor and the direction vector of the box.
    """

    def __init__(self, d, theta_v_p_nn_v_p_c, l, w, theta_v_p_nn_v_dir):
        self.d = d
        self.theta_v_p_nn_v_p_c = theta_v_p_nn_v_p_c
        self.l = l
        self.w = w
        self.theta_v_p_nn_v_dir = theta_v_p_nn_v_dir

    def get_absolute_bounding_box(self, bb_point_coord, nn_point_coord):

        relative_rotated_bb = self.rotation_invariant_representation_to_relative_rotated_bb(
            bb_point_coord, nn_point_coord)
        absolute_bounding_box = relative_rotated_bb.get_absolute_bounding_box(
            bb_point_coord[0], bb_point_coord[1])
        return absolute_bounding_box

    def rotation_invariant_representation_to_relative_rotated_bb(self, bb_point_coord: np.ndarray, nn_point_coord: np.ndarray) -> RelativeBoundingBox:
        """ Transforms the rotation invariant bounding box to an non rotation invariant representation defined relative to the corresponding radar point.

        Args:
            bb_point_coord: Spatial coordinates of the radar point to which the bounding box belongs.
            nn_point_coord: Spatial coordinates of the nearest neighbor of the radar point.

        Returns:
            relative_rotated_bb: Relative, rotated bounding box of the form [x_center, y_center, length, width, theta]
                - x_center: x-position of the bounding box center relative to the radar detection it corresponds to.
                - y_center: y-position of the bounding box center relative to the radar detection it corresponds to.
        """
        # distance vector: point -> nearest neighbor
        v_p_nn = (nn_point_coord - bb_point_coord).reshape(2, 1)

        # recover angle between v_dir and x-axis
        # 1. angle between v_p_nn and x-axis
        v_p_nn_norm = v_p_nn / np.linalg.norm(v_p_nn)
        theta_v_p_nn_x = atan2(
            v_p_nn_norm[1, 0], v_p_nn_norm[0, 0]) * 180 / np.pi

        # 2. clockwise angle between v_p_nn and v_dir
        self.theta_v_p_nn_v_dir = self.theta_v_p_nn_v_dir

        # 3. recovered angle between v_dir and x-axis
        theta_v_dir_x = np.round(self.theta_v_p_nn_v_dir + theta_v_p_nn_x, 5)

        # 5. get angle in range 0 - 180 (theta_v_dir_x x° = 180° + x° = -180° + x°)
        while theta_v_dir_x < 0:
            theta_v_dir_x = 360 + theta_v_dir_x

        while theta_v_dir_x >= 180:
            theta_v_dir_x = theta_v_dir_x - 180

        # recover relative position of bounding box center
        # 1. angle between v_p_nn and x-axis
        theta_v_p_nn_x = theta_v_p_nn_x

        # 2. clockwise angle from v_p_nn to v_p_c
        self.theta_v_p_nn_v_p_c = self.theta_v_p_nn_v_p_c

        # 3. recovered angle between v_p_c and x-axis
        theta_v_p_c_x = self.theta_v_p_nn_v_p_c + theta_v_p_nn_x

        # 4. to not allow angles > 360
        while theta_v_p_c_x > 360:
            theta_v_p_c_x = theta_v_p_c_x - 360

        # 5. transform cylinder to cartesian coordinates
        x_center = self.d * np.cos((theta_v_p_c_x * np.pi) / 180)
        y_center = self.d * np.sin((theta_v_p_c_x * np.pi) / 180)

        # but everything together in bb encoding
        relative_rotated_bb = RelativeRotatedBoundingBox(
            x_center, y_center, self.l, self.w, theta_v_dir_x)

        return relative_rotated_bb


class RelativeRotatedBoundingBox(RelativeBoundingBox):
    """ Relative rotated bounding box.

    Attributes:
        x_center: Relative x-position of the bounding box center to the radar point it corresponds to.
        y_center: Relative y-position of the bounding box center to the radar point it corresponds to.
        l: Length of the box.
        w: Width of the box.
        theta: Angle between longitudinal box-axis and x-axis of the coordinate system.
    """

    def __init__(self, x_center, y_center, l, w, theta):
        self.x_center = x_center
        self.y_center = y_center
        self.l = l
        self.w = w
        self.theta = theta

    def get_absolute_bounding_box(self, x: float, y: float):

        # corners relative to zero origin - long side along x axis
        c1_orig = np.array([self.l / 2, self.w / 2]).reshape(1, 2)
        c2_orig = np.array([self.l / 2, - self.w / 2]).reshape(1, 2)
        c3_orig = np.array([- self.l / 2, - self.w / 2]).reshape(1, 2)
        c4_orig = np.array([- self.l / 2, self.w / 2]).reshape(1, 2)
        corners_orig = np.concatenate(
            (c1_orig, c2_orig, c3_orig, c4_orig), axis=0)

        # rotate bounding box corners by theta
        theta_x = self.theta
        theta_rad = (theta_x * np.pi) / 180
        rot = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                        [np.sin(theta_rad), np.cos(theta_rad)]])

        corners_orig = np.matmul(rot, corners_orig.T).T

        # absolute center of bounding box using point coordinates
        c_abs = np.array([x, y]).reshape(
            1, 2) + np.array([self.x_center, self.y_center]).reshape(1, 2)

        # translate (rotated) corners to get absolute position of bb corners
        c1_bb = corners_orig[0, :].reshape(1, 2) + c_abs
        c2_bb = corners_orig[1, :].reshape(1, 2) + c_abs
        c3_bb = corners_orig[2, :].reshape(1, 2) + c_abs
        c4_bb = corners_orig[3, :].reshape(1, 2) + c_abs
        corners = np.concatenate((c1_bb, c2_bb, c3_bb, c4_bb), axis=0)

        return BoundingBox(corners, False)

    def relative_rotated_bb_to_rotation_invariant_representation(self, bb_point_coord, nn_point_coord) -> RotationInvariantRelativeRotatedBoundingBox:
        """ Transforms the relative rotated bounding box to an rotation invariant representation.

        Args:
            bb_point_coord: Spatial coordinates of the radar point to which the bounding box belongs.
            nn_point_coord: Spatial coordinates of the nearest neighbor of the radar point.

        Returns:
            bb_rot_inv: Rotation invariant bounding box of the form [d, theta_v_p_nn_v_p_c, length, width, theta_v_p_nn_v_dir]
                - d: Distance between the point and the box center.
                - theta_v_p_nn_v_p_c: Clockwise angle between the vector point->nearest-neighbor and the vector point->bounding-box-center.
                - theta_v_p_nn_v_dir: Clockwise angle between the vector point->nearest-neighbor and the direction vector of the box.
        """

        # distance vector: point -> nearest neighbor
        v_p_nn = (nn_point_coord - bb_point_coord).reshape(2, 1)

        # distance vector: point -> center
        v_p_c = np.array([self.x_center, self.y_center]).reshape(2, 1)

        # long bounding box direction vector
        v_dir = np.array([1, np.tan((self.theta * np.pi) / 180)]).reshape(2, 1)

        # clockwise angle from v_p_nn to v_dir:
        # 1. get angle between v_dir and x-axis in range [-180, 180]
        v_dir_norm = v_dir / np.linalg.norm(v_dir)
        theta_v_dir_x = atan2(v_dir_norm[1, 0], v_dir_norm[0, 0]) * 180 / np.pi

        # 2. get angle between v_p_nn and x-axis in range [-180, 180]
        v_p_nn_norm = v_p_nn / np.linalg.norm(v_p_nn)
        theta_v_p_nn_x = atan2(
            v_p_nn_norm[1, 0], v_p_nn_norm[0, 0]) * 180 / np.pi

        # 3. angle from v_p_nn to v_dir based on difference of angles to x Axis
        theta_v_p_nn_v_dir = np.round(theta_v_dir_x - theta_v_p_nn_x, 5)
        while theta_v_p_nn_v_dir < 0:
            theta_v_p_nn_v_dir = 360 + theta_v_p_nn_v_dir

        # 5. get angle only in range 0 - 180
        while theta_v_p_nn_v_dir >= 180:
            theta_v_p_nn_v_dir = theta_v_p_nn_v_dir - 180

        # clockwise angle from v_p_nn to v_p_c:
        if np.linalg.norm(v_p_c) == 0:
            # if point itself is center of the bb
            theta_v_p_nn_v_p_c = 0
        else:
            # 1. get angle between v_p_c and x-axis in range [-180, 180]
            v_p_c_norm = v_p_c / np.linalg.norm(v_p_c)
            theta_v_p_c_x = atan2(
                v_p_c_norm[1, 0], v_p_c_norm[0, 0]) * 180 / np.pi

            # 2. get angle between v_p_nn and x-axis in range [-180, 180]
            theta_v_p_nn_x = theta_v_p_nn_x

            # 3. angle from v_p_nn to v_p_c based on difference of angles to x Axis
            theta_v_p_nn_v_p_c = np.round(theta_v_p_c_x - theta_v_p_nn_x, 5)
            while theta_v_p_nn_v_p_c < 0:
                theta_v_p_nn_v_p_c = 360 + theta_v_p_nn_v_p_c

        # distance between bb center and point
        d = np.linalg.norm(v_p_c)

        # but everything together in bb encoding
        bb_rot_inv = RotationInvariantRelativeRotatedBoundingBox(
            d, theta_v_p_nn_v_p_c, self.l, self.w, theta_v_p_nn_v_dir)

        return bb_rot_inv


class RelativeAlignedBoundingBox(RelativeBoundingBox):
    """ Relative aligned bounding box.

    Attributes:
        x_center: Relative x-position of the bounding box center to the radar point it corresponds to.
        y_center: Relative y-position of the bounding box center to the radar point it corresponds to.
        dx: Length of the box.
        dy: Width of the box.
    """

    def __init__(self, x_center, y_center, dx, dy):
        self.x_center = x_center
        self.y_center = y_center
        self.dx = dx
        self.dy = dy

    def get_absolute_bounding_box(self, x: float, y: float):

        # corners relative to zero origin
        c1_orig = np.array([self.dx / 2, self.dy / 2]).reshape(1, 2)
        c2_orig = np.array([self.dx / 2, - self.dy / 2]).reshape(1, 2)
        c3_orig = np.array([- self.dx / 2, - self.dy / 2]).reshape(1, 2)
        c4_orig = np.array([- self.dx / 2, self.dy / 2]).reshape(1, 2)
        corners_orig = np.concatenate(
            (c1_orig, c2_orig, c3_orig, c4_orig), axis=0)

        # absolute center of bounding box using point coordinates
        c_abs = np.array([x, y]).reshape(
            1, 2) + np.array([self.x_center, self.y_center]).reshape(1, 2)

        # translate (rotated) corners to get absolute position of bb corners
        c1_bb = corners_orig[0, :].reshape(1, 2) + c_abs
        c2_bb = corners_orig[1, :].reshape(1, 2) + c_abs
        c3_bb = corners_orig[2, :].reshape(1, 2) + c_abs
        c4_bb = corners_orig[3, :].reshape(1, 2) + c_abs
        corners = np.concatenate((c1_bb, c2_bb, c3_bb, c4_bb), axis=0)

        return BoundingBox(corners, True)


class BoundingBox():
    """ General bounding box representation.

    Attributes:
        corners: Matrix with the absolute coordinates of the four bounding box corners.
        is_aligned: Bool whether box is aligned or not.
        is_rotated: Bool whether box is rotated or not.

    Methods:
        get_relative_bounding_box: Transforms the box to a representation, defined relative to the corresponding radar point.
        get_to_two_point_representation: Transforms the box to the format [x_min, y_min, x_max, y_max] used for NMS.
        plot_on_axis: Plot the box on a given axis.
        get_two_point_representations: Transforms multiple boxes to the format [x_min, y_min, x_max, y_max] used for NMS aligned.
        get_absolute_rotated_box_representations: Transforms multiple boxes to the format [x, y, l, w, theta] used for NMS rotated.
    """
    def __init__(self, corners: np.ndarray, aligned: bool):
        self.corners = corners
        self.is_aligned = aligned
        self.is_rotated = not aligned

    def get_relative_bounding_box(self, x: float, y: float) -> RelativeBoundingBox:
        if self.is_aligned and not self.is_rotated:
            bb = self.__get_relative_aligned_bounding_box(x, y)

        elif self.is_rotated and not self.is_aligned:
            bb = self.__get_relative_rotated_bounding_box(x, y)

        return bb

    def __get_relative_rotated_bounding_box(self, x: float, y: float) -> RelativeRotatedBoundingBox:

        # get length and width
        p1 = self.corners[0, :].reshape(1, 2)
        p2 = self.corners[1, :].reshape(1, 2)
        p3 = self.corners[2, :].reshape(1, 2)
        p4 = self.corners[3, :].reshape(1, 2)

        d1 = np.linalg.norm(p1 - p2)
        d2 = np.linalg.norm(p1 - p3)
        d3 = np.linalg.norm(p1 - p4)

        # smallest entry is width, middle one is length, biggest is diagonal
        d = [d1, d2, d3]
        w = min(d)
        d.remove(w)
        l = min(d)

        # get center of the bounding box
        c = ((p1 + p2 + p3 + p4) / 4).reshape(1, 2)

        # get direction vector of the longer side
        if l == np.linalg.norm(p1 - p2):
            v_l = (p1 - p2).reshape(2, 1)

        elif l == np.linalg.norm(p1 - p3):
            v_l = (p1 - p3).reshape(2, 1)

        elif l == np.linalg.norm(p1 - p4):
            v_l = (p1 - p4).reshape(2, 1)

        else:
            v_l = np.zeros([2, 1])
            raise Exception("No longest side found")

        # get angle between longe side direction vector and x-axis in range [-180, 180]
        v_l_norm = v_l / np.linalg.norm(v_l)
        theta_x = atan2(v_l_norm[1, 0], v_l_norm[0, 0]) * 180 / np.pi

        # transform to angle between 0-180 -> e.g. -45° and 135° is same bounding box
        if theta_x < 0:
            theta_x = 180 + theta_x

        # get relative position of bb center w.r.t point
        relative_position = c - np.array([x, y]).reshape(1, 2)

        x_rel = relative_position[0, 0]
        y_rel = relative_position[0, 1]

        theta_rel = theta_x
        return RelativeRotatedBoundingBox(x_rel, y_rel, l, w, theta_rel)

    def __get_relative_aligned_bounding_box(self, x: float, y: float) -> RelativeAlignedBoundingBox:
        # get length and width
        p1 = self.corners[0, :].reshape(1, 2)
        p2 = self.corners[1, :].reshape(1, 2)
        p3 = self.corners[2, :].reshape(1, 2)
        p4 = self.corners[3, :].reshape(1, 2)

        # get center of the bounding box
        c = ((p1 + p2 + p3 + p4) / 4).reshape(1, 2)

        # get size in x and y direction (based on created points in function "minimum_bounding_rectangle_without_rotation")
        dx = abs(p1[0, 0] - p3[0, 0])
        dy = abs(p1[0, 1] - p2[0, 1])

        # get relative position of bb center w.r.t point
        relative_position = c - np.array([x, y]).reshape(1, 2)

        x_rel = relative_position[0, 0]
        y_rel = relative_position[0, 1]

        return RelativeAlignedBoundingBox(x_rel, y_rel, dx, dy)

    def get_to_two_point_representation(self) -> np.ndarray:
        """ Returns bounding box definition in the form: [x_min, y_min, x_max, y_max]

        Note: Format required for torchvisions NMS
        """
        x_min = np.min(self.corners[:, 0])
        x_max = np.max(self.corners[:, 0])
        y_min = np.min(self.corners[:, 1])
        y_max = np.max(self.corners[:, 1])

        return np.array([x_min, y_min, x_max, y_max])

    def plot_on_axis(self, ax):
        # draw bounding box corners, center and edges
        # ax.scatter(self.corners[:,0], self.corners[:,1], c = "red", marker = "x")
        ax.plot([self.corners[0, 0], self.corners[1, 0]], [
                self.corners[0, 1], self.corners[1, 1]], c="black")
        ax.plot([self.corners[0, 0], self.corners[2, 0]], [
                self.corners[0, 1], self.corners[2, 1]], c="black")
        ax.plot([self.corners[0, 0], self.corners[3, 0]], [
                self.corners[0, 1], self.corners[3, 1]], c="black")
        ax.plot([self.corners[1, 0], self.corners[2, 0]], [
                self.corners[1, 1], self.corners[2, 1]], c="black")
        ax.plot([self.corners[1, 0], self.corners[3, 0]], [
                self.corners[1, 1], self.corners[3, 1]], c="black")
        ax.plot([self.corners[2, 0], self.corners[3, 0]], [
                self.corners[2, 1], self.corners[3, 1]], c="black")

    @staticmethod
    def get_two_point_representations(bounding_boxes: list) -> np.ndarray:
        """ Transforms a list of BoundingBox objects to a matrix of its two point representations.

        Note: This format is required for nms of torchivison

        Args:
            bounding_boxes: List of instances of the class BoundingBox.

        Returns:
            bounding_box_matrix: Matrix with each row containing a bounding box in the format: [x_min, y_min, x_max, y_max]
        """
        bounding_box_matrix = np.empty([len(bounding_boxes), 4])

        for i, box in enumerate(bounding_boxes):
            box_two_point = box.get_to_two_point_representation()
            bounding_box_matrix[i, :] = box_two_point

        return bounding_box_matrix

    @staticmethod
    def get_absolute_rotated_box_representations(bounding_boxes: list) -> np.ndarray:
        """ Transforms a list of BoundingBox objects to a matrix of absolute rotated bounding boxes.

        Note: This format is required for nms_rotated of detectron2

        Args:
            bounding_boxes: List of instances of the class BoundingBox.

        Returns:
            bounding_box_matrix: Matrix with each row containing a bounding box in the format: [x, y, l, w, theta]
        """
        # bring the bounding boxes in the format required by rotated_nms
        bounding_box_matrix = np.empty([len(bounding_boxes), 5])

        for i, box in enumerate(bounding_boxes):

            # get length and width
            p1 = box.corners[0, :].reshape(1, 2)
            p2 = box.corners[1, :].reshape(1, 2)
            p3 = box.corners[2, :].reshape(1, 2)
            p4 = box.corners[3, :].reshape(1, 2)

            d1 = np.linalg.norm(p1 - p2)
            d2 = np.linalg.norm(p1 - p3)
            d3 = np.linalg.norm(p1 - p4)

            # smallest entry is width, middle one is length, biggest is diagonal
            d = [d1, d2, d3]
            w = min(d)
            d.remove(w)
            l = min(d)

            # get center of the bounding box
            c = ((p1 + p2 + p3 + p4) / 4).reshape(1, 2)

            # get direction vector of the longer side
            if l == np.linalg.norm(p1 - p2):
                v_l = (p1 - p2).reshape(2, 1)

            elif l == np.linalg.norm(p1 - p3):
                v_l = (p1 - p3).reshape(2, 1)

            elif l == np.linalg.norm(p1 - p4):
                v_l = (p1 - p4).reshape(2, 1)

            else:
                bounding_box_matrix[i, :] = np.array([0, 0, 1, 1, 0]).reshape(1, 5)
                Warning("Invalid bounding box for one node - using default bounding box")
                continue

            # get angle between longe side direction vector and x-axis in range [-180, 180]
            v_l_norm = v_l / np.linalg.norm(v_l)
            theta_x = atan2(v_l_norm[1, 0], v_l_norm[0, 0]) * 180 / np.pi

            # transform to angle between 0-180 -> e.g. -45° and 135° is same bounding box
            if theta_x < 0:
                theta_x = 180 + theta_x

            bb = np.array([c[0, 0], c[0, 1], l, w, theta_x]).reshape(1, 5)
            bounding_box_matrix[i, :] = bb

        return bounding_box_matrix


def absolute_rotated_box_representation_to_boundingbox(x, y, l, w, theta) -> BoundingBox:
    corners = get_box_corners(x, y, l, w, theta)
    return BoundingBox(corners, False)


def adapt_bb_orientation_angle(bb_matrix: np.ndarray):
    """ Adapts the definition of the orientation angel "theta" of rotated bounding boxes.

    Changes the angle range of theta from [0, pi] to the range [-pi/2, pi/2]
    Applies a sin function, to map the angles then from the range [-pi/2, pi/2] to the range [-1, 1].

    Args:
        bb_matrix: Matrix with each row containing a rotated bounding box. Angle of each box in range [0, pi].

    Returns:
        bb_matrix: Matrix with each row containing a rotated bounding box. Angle of each box in range [-1, 1].
    """

    # iterate through all rows which contain boxes -> no 'nan' entry
    for i in range(bb_matrix.shape[0]):
        if not np.isnan(bb_matrix[i, 0]):

            theta_x_l = bb_matrix[i, 4]

            # flip angles above pi/2 to get angles in range  [-pi/2, pi/2]
            theta_shift = theta_x_l - np.pi if theta_x_l > np.pi / 2 else theta_x_l

            # apply sin to map angles from to range [-1, 1]
            theta_smooth = np.sin(theta_shift)

            bb_matrix[i, 4] = theta_smooth

    return bb_matrix


def invert_bb_orientation_angle_adaption(theta_x_l):
    """ Inverts the orientation angel "theta" of rotated bounding boxes to its original format.

    Applies an arcsin to get the angle range of theta from [-1, 1] to the range [-pi/2, pi/2].
    Then maps the angles back from the range [-pi/2, pi/2] to the range [0, pi].

    Args:
        bb_matrix: Matrix with each row containing a rotated bounding box. Angle of each box in range [-1, 1].

    Returns:
        bb_matrix: Matrix with each row containing a rotated bounding box. Angle of each box in range [0, pi].
    """

    # arcsin is only defined in range [-1, 1]
    # if theta_x_l is out of this range -> set it to limit of the range
    theta_x_l = max(min(theta_x_l, 1), -1)

    # invert the sin
    theta_unsmoothed = np.arcsin(theta_x_l)

    # bring angle back in range [0 - pi]
    theta_rec = theta_unsmoothed + np.pi if theta_unsmoothed < 0 else theta_unsmoothed

    return theta_rec
