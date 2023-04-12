import numpy as np
import matplotlib.pyplot as plt

from radar_scenes.sensors import get_mounting
from radar_scenes.coordinate_transformation import transform_detections_sequence_to_car
from radar_scenes.labels import ClassificationLabel

from radargnn.preprocessor.radar_point_cloud import RadarPointCloud
from radargnn.utils.radar_scenes_properties import Colors


class SceneCollection():
    """ Implements a collection of the data of multiple consecutive scenes of the RadarScenes dataset.

    The data of these consecutive scenes can then be transformed to a single point cloud frame.

    Attributes:
        scenes: List of consecutive scenes.
        point_cloud: Point cloud, created from the scenes.

    Methods:
        process: Extracts and transforms the data from the scenes.
        extract_scene_radar_data: Extracts data from consecutive scenes into a point cloud instance.
        transform_radar_data: Transforms data from consecutive scenes.
        show: Visualizes the data from the scenes.
    """

    def __init__(self):
        self.scenes = []
        self.point_cloud = None

    def process(self, use_reduced_classes=True) -> None:
        self.extract_scene_radar_data(use_reduced_classes)
        self.transform_radar_data()

    def extract_scene_radar_data(self, use_reduced_classes: bool = True) -> None:
        """ Extracts the radar data of consecutive scenes and stores them as point cloud.

        Args:
            use_reduced_classes: Decides to use either 11 or the 5 main object classes of the RadarScenes dataset.
        """
        point_cloud = RadarPointCloud()

        # get number of points
        num_points = sum((len(scene.radar_data) for scene in self.scenes))

        # initialize storage for Point Cloud
        point_cloud.X_cc = np.empty([num_points, 2])
        point_cloud.X_seq = np.empty([num_points, 2])
        point_cloud.V_cc = np.empty([num_points, 2])
        point_cloud.V_cc_compensated = np.empty([num_points, 2])
        point_cloud.range_sc = np.empty([num_points, 1])
        point_cloud.azimuth_sc = np.empty([num_points, 1])
        point_cloud.rcs = np.empty([num_points, 1])
        point_cloud.vr = np.empty([num_points, 1])
        point_cloud.vr_compensated = np.empty([num_points, 1])
        point_cloud.timestamp = np.empty([num_points, 1])
        point_cloud.sensor_id = np.empty([num_points, 1])
        point_cloud.uuid = []
        point_cloud.track_id = []

        point_cloud.label_id = np.empty([num_points, 1])

        # store the radar data of all scenes in the point cloud

        count = 0
        for scene in self.scenes:
            for point in scene.radar_data:
                point_cloud.timestamp[count] = point[0]
                point_cloud.sensor_id[count] = point[1]
                point_cloud.range_sc[count] = point[2]
                point_cloud.azimuth_sc[count] = point[3]
                point_cloud.rcs[count] = point[4]
                point_cloud.vr[count] = point[5]
                point_cloud.vr_compensated[count] = point[6]
                point_cloud.X_cc[count, 0] = point[7]
                point_cloud.X_cc[count, 1] = point[8]
                point_cloud.X_seq[count, 0] = point[9]
                point_cloud.X_seq[count, 1] = point[10]
                point_cloud.uuid.append(point[11])
                point_cloud.track_id.append(point[12])
                if use_reduced_classes:
                    clabel = ClassificationLabel.label_to_clabel(point[13])
                    if clabel is not None:
                        point_cloud.label_id[count] = clabel.value
                    else:
                        point_cloud.label_id[count] = None
                else:
                    point_cloud.label_id[count] = point[13]
                count += 1

        """
        TODO
        Improved version with only one loop and not looping over points
        -> Not yet working properly

        count = 0
        for scene in self.scenes:
            size = scene.radar_data.size
            points = scene.radar_data[count:size]

            point_cloud.timestamp[count:size,0] = points["timestamp"]
            point_cloud.sensor_id[count:size,0] = points["sensor_id"]
            point_cloud.range_sc[count:size,0] = points["range_sc"]
            point_cloud.azimuth_sc[count:size,0] = points["azimuth_sc"]
            point_cloud.rcs[count:size,0] = points["rcs"]
            point_cloud.vr[count:size,0] = points["vr"]
            point_cloud.vr_compensated[count:size,0] = points["vr_compensated"]
            point_cloud.X_cc[count:size,0] = points["x_cc"]
            point_cloud.X_cc[count:size,1] = points["y_cc"]
            point_cloud.X_seq[count:size,0] = points["x_seq"]
            point_cloud.X_seq[count:size,1] = points["y_seq"]
            point_cloud.uuid.append(points["uuid"])
            point_cloud.track_id.append(points["track_id"])
            point_cloud.label_id[count:size,0] = points["label_id"]

            # if use_reduced_classes:
            #     clabel = ClassificationLabel.label_to_clabel(points["label_id"])
            #     if clabel is not None:
            #         point_cloud.label_id[count:size,0] = clabel.value
            #     else:
            #         point_cloud.label_id[count:size,0] = None
            # else:
            #     point_cloud.label_id[count:size,0] = points["label_id"]

            count += scene.radar_data.size
        """

        self.point_cloud = point_cloud

    def transform_radar_data(self) -> None:
        """ Processes radar data.

        Applies the following transformations:
            - Calculate x,y position of the sequences in car coordinate system of the first scene
            - Calculate and saves the velocities in x,y direction
        """

        # transform Points from sequence-KoSy to car-KoSy of scene scene in the collection
        x, y = transform_detections_sequence_to_car(
            self.point_cloud.X_seq[:, 0], self.point_cloud.X_seq[:, 1], self.scenes[0].odometry_data)
        self.point_cloud.X_seq = np.stack((x, y), axis=-1)

        # get the V_cc/V_cc_compensated (=[v_x, v_y]) value based on v_r/vr_compensated and azimuth_sc
        '''
        TODO:
        - Use the mounting of the json file and not the default mounting
        '''
        sensor_yaw = np.array([get_mounting(s_id[0], json_path=None)[
                              "yaw"] for s_id in self.point_cloud.sensor_id])
        angles = self.point_cloud.azimuth_sc + sensor_yaw.reshape(sensor_yaw.shape[0], 1)

        self.point_cloud.V_cc = np.concatenate(
            [self.point_cloud.vr * np.cos(angles), self.point_cloud.vr * np.sin(angles)], axis=1)
        self.point_cloud.V_cc_compensated = np.concatenate(
            [self.point_cloud.vr_compensated * np.cos(angles), self.point_cloud.vr_compensated * np.sin(angles)], axis=1)

    def show(self) -> None:

        # get the car position and velocity in car KoSy
        x_car_seq = 0
        y_car_seq = 0
        v_x_car = self.scenes[0].odometry_data[4]
        v_y_car = 0

        # get the camera image of the scene
        img_path = self.scenes[0].camera_image_name
        img = plt.imread(img_path)

        # convert label IDs to colors
        colors = Colors()
        c = [colors.label_id_to_color[id[0]] for id in self.point_cloud.label_id]

        # create and show plot
        _, (ax, ax2) = plt.subplots(1, 2)
        ax.scatter(self.point_cloud.X_cc[:, 0], self.point_cloud.X_cc[:, 1], c=c)
        ax.scatter(x_car_seq, y_car_seq, c='black')
        ax.quiver(x_car_seq, y_car_seq, v_x_car, v_y_car, scale=150)
        ax.quiver(self.point_cloud.X_cc[:, 0], self.point_cloud.X_cc[:, 1],
                  self.point_cloud.V_cc_compensated[:, 0], self.point_cloud.V_cc_compensated[:, 1], scale=150)
        ax.axis("equal")
        ax2.imshow(img)


def concatenate_subsequent_scenes(sequence, start_timestamp, time) -> SceneCollection:
    """ Collects consecutive scenes of one RadarScenes sequences and stores them into a SceneCollection object.

    Args:
        sequence: Sequence of the RadarScenes dataset to process.
        start_timestamp: Timestamp of the first scene.
        time: Time duration for collecting subsequent scenes.

    Returns:
        scene_collection: SceneCollection object with a list of the collected consecutive scenes.
    """

    first_scene = sequence.get_scene(start_timestamp)
    scene_collection = SceneCollection()

    # All code below may be replaced by this line, BUT it is still buggy for some special cases!
    # -> First find the bug before replacing the old working longer version with this oneliner!!!

    # scene_collection.scenes = [sequence.get_scene(t) for t in filter(
    # lambda t: 0 <= (t - first_scene.timestamp) * 1e-6 < time, sequence.timestamps)]

    subsuq_scene = sequence.next_scene_after(start_timestamp)
    scene_collection.scenes.append(first_scene)

    if subsuq_scene is not None:
        scene_collection.scenes.append(subsuq_scene)
        current_timestamp = start_timestamp

        while ((subsuq_scene.timestamp - first_scene.timestamp) * 1e-6 < time):

            # get the next time stamp and corresponding scene data
            current_timestamp = sequence.next_timestamp_after(current_timestamp)

            if current_timestamp is not None:
                subsuq_scene = sequence.next_scene_after(current_timestamp)

                if subsuq_scene is not None:
                    # add the following scene to the scene collection
                    scene_collection.scenes.append(subsuq_scene)
                else:
                    break

            else:
                break

    return scene_collection
