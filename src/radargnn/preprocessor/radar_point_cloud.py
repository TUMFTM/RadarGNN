import matplotlib.pyplot as plt
import numpy as np

from radargnn.utils.radar_scenes_properties import Colors


class RadarPointCloud():
    """ Point cloud containing all features / radar channels of the RadarScenes dataset.

    Methods:
        remove_points_without_labelID: Removes all points without a valid label ID.
        remove_points_without_valid_velocity: Removes all points without doppler velocity.
        remove_points_out_of_range: Removes points based on their distance to the car-coordinate-system origin.
        remove_points_based_on_index: Removes points defined by their index.
        show: Visualize the point cloud.
    """

    def __init__(self):
        self.X_cc = None
        self.X_seq = None

        self.V_cc = None
        self.V_cc_compensated = None

        self.range_sc = None
        self.azimuth_sc = None
        self.rcs = None

        self.vr = None
        self.vr_compensated = None

        self.timestamp = None
        self.sensor_id = None

        self.uuid = None
        self.track_id = None
        self.label_id = None

    def remove_points_without_labelID(self) -> None:
        """ Removes all points with a "non" class label.

        Note:
            This results from using reduced labels (clabel) where e.g. the class "animal" becomes none
        """
        idx_rmv = np.where(np.isnan(self.label_id[:, 0]))[0]
        self.remove_points_based_on_index(idx_rmv)

    def remove_points_without_valid_velocity(self) -> None:
        """ Removes all points with a "non" velocity.

        Note:
            This results from using reduced labels (clabel) where e.g. the velocity of "animal" becomes none
        """
        idx_rmv_1 = np.where(np.isnan(self.V_cc_compensated[:, 0]))[0]
        idx_rmv_2 = np.where(np.isnan(self.V_cc_compensated[:, 1]))[0]
        idx_rmv = np.unique(np.concatenate((idx_rmv_1, idx_rmv_2), axis=0))

        self.remove_points_based_on_index(idx_rmv)

    def remove_points_out_of_range(self, x_max: float, y_max: float) -> None:
        """ Removes radar points based on their location.

        Removes all points that are:
            - further away than x_max or y_max from the car
            - behind the car -> x < 0
        """

        idx_inv1 = np.where(abs(self.X_cc[:, 1]) > y_max)[0]
        idx_inv2 = np.where(self.X_cc[:, 0] > x_max)[0]
        idx_inv3 = np.where(self.X_cc[:, 0] < 0)[0]
        idx_inv = np.concatenate([idx_inv1, idx_inv2, idx_inv3], axis=0)
        idx_inv = np.unique(idx_inv)
        self.remove_points_based_on_index(idx_inv)

    def remove_points_based_on_index(self, idx_array: np.ndarray) -> None:
        """ Remove points by their index.
        """

        for key in vars(self).keys():
            if vars(self).get(key) is not None:
                vars(self)[key] = np.delete(vars(self)[key], idx_array, axis=0)

    def show(self, show_velocity_vector=False) -> None:

        # convert label IDs to colors
        colors = Colors()
        c = [colors.label_id_to_color[id[0]] for id in self.label_id]

        # create and show plot
        fig, ax = plt.subplots()
        ax.scatter(self.X_cc[:, 0], self.X_cc[:, 1], c=c)
        ax.scatter(0, 0, c='black')
        if show_velocity_vector:
            ax.quiver(self.X_cc[:, 0], self.X_cc[:, 1], self.V_cc_compensated[:, 0], self.V_cc_compensated[:, 1], scale=150)
        ax.axis("equal")

        return fig, ax
