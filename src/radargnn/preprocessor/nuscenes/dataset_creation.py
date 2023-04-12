import os
import json
from dataclasses import asdict
from typing import List

import torch
from torch_geometric.data import Data, Dataset

import numpy as np

from nuscenes import nuscenes
from pyquaternion.quaternion import Quaternion

from radargnn.preprocessor.configs import GraphConstructionConfiguration
from radargnn.preprocessor.nuscenes import conversion
from radargnn.preprocessor.nuscenes import utils
from radargnn.preprocessor.nuscenes.configs import NuScenesDatasetConfiguration, NuScenesSplitConfiguration


class NuScenesGraphDataset(Dataset):
    def __init__(self,
                 root: str,
                 graph_config: GraphConstructionConfiguration,
                 dataset_config: NuScenesDatasetConfiguration,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        # to be defined before calling super
        self.graph_config = graph_config
        self.dataset_config = dataset_config

        # initialize nuscenes graph dataset parameters
        self.nsweeps = self.dataset_config.nsweeps
        self.wlh_factor = self.dataset_config.wlh_factor
        self.wlh_offset = self.dataset_config.wlh_offset

        # super constructor executes "download" and/or "process" method
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        # no download of the raw dataset if this file is found in "root/raw" folder
        file_list = ['License.md']
        return file_list

    @property
    def processed_file_names(self):
        # no download if this file is found in "root/processed" folder
        file_list = ['config.json']
        return file_list

    def download(self):
        raise NotImplementedError(
            'Data must be downloaded manually and stored in the'
            f'{self.root}/raw folder!'
        )

    @staticmethod
    def _get_box_label(name: str) -> int:
        """ Returns the nuScenes detection challenge class for a given class name.

        Converts the class names of the original 23 classes of the nuScenes dataset
        to a numeric value corresponding with the 10 classes of the nuScenes detection
        challenge classes.

        Arguments:
            name: Class name of the nuScenes class.

        Returns:
            Class id for the nuScenes detection challenge.
        """
        class_name_to_id = {
            'animal': 0,
            'human.pedestrian.personal_mobility': 0,
            'human.pedestrian.stroller': 0,
            'human.pedestrian.wheelchair': 0,
            'movable_object.debris': 0,
            'movable_object.pushable_pullable': 0,
            'static_object.bicycle_rack': 0,
            'vehicle.emergency.ambulance': 0,
            'vehicle.emergency.police': 0,
            'movable_object.barrier': 1,
            'vehicle.bicycle': 2,
            'vehicle.bus.bendy': 3,
            'vehicle.bus.rigid': 3,
            'vehicle.car': 4,
            'vehicle.construction': 5,
            'vehicle.motorcycle': 6,
            'human.pedestrian.adult': 7,
            'human.pedestrian.child': 7,
            'human.pedestrian.construction_worker': 7,
            'human.pedestrian.police_officer': 7,
            'movable_object.trafficcone': 8,
            'vehicle.trailer': 9,
            'vehicle.truck': 10
        }
        return class_name_to_id[name]

    @staticmethod
    def filter_bounding_boxes(nusc, boxes):
        """Removes all bounding boxes without associated radar or lidar points.

        Similar to the filter method applied during the official nuScenes evaluation.

        Arguments:
            nusc: NuScenes instance.
            boxes: List of bounding boxes.

        Returns:
            boxes: Filtered list of bounding boxes.
        """
        # Get corresponding annotations
        annotations = [nusc.get('sample_annotation', box.token) for box in boxes]

        # Filter bounding boxes
        boxes = [box for box, annotation in zip(boxes, annotations) if annotation['num_lidar_pts'] + annotation['num_radar_pts'] > 0]

        return boxes

    def crop_bounding_boxes(self, boxes):
        """ Crops the given bounding boxes according to the crop settings.

        Arguments:
            points: Original bounding boxes.

        Returns:
            points: Cropped bounding boxes.
        """
        # Initialize boundary values
        xlim = self.dataset_config.crop_settings['x']
        ylim = self.dataset_config.crop_settings['y']

        # Filter points along x-dimension
        boxes = [box for box in boxes if (-xlim < box.center[0] < xlim) & (-ylim < box.center[1] < ylim)]

        return boxes

    def crop_point_cloud(self, points):
        """ Crops the given point cloud according to the crop settings.

        Arguments:
            points: Original point cloud.

        Returns:
            points: Cropped point cloud.
        """
        # Initialize boundary values
        xlim = np.full_like(points[0, :], fill_value=self.dataset_config.crop_settings['x'])
        ylim = np.full_like(points[1, :], fill_value=self.dataset_config.crop_settings['y'])

        # Filter points along x-dimension
        upper_x_mask = np.greater(points[0, :], xlim)
        lower_x_mask = np.less(points[0, :], -xlim)

        # Filter points along y-dimension
        upper_y_mask = np.greater(points[1, :], ylim)
        lower_y_mask = np.less(points[1, :], -ylim)

        # Combine masks
        x_mask = np.logical_or(upper_x_mask, lower_x_mask)
        y_mask = np.logical_or(upper_y_mask, lower_y_mask)
        mask = np.logical_or(x_mask, y_mask)

        return points[:, ~mask]

    def get_sensor_points(self, nusc, sample: dict, sensor: str) -> np.ndarray:
        """ Returns the radar points belonging to a given sample and sensor.

        Arguments:
            nusc: nuScenes instance of the selected dataset version.
            sample: Single nuScenes sample instance.
            sensor: Single radar sensor instance.

        Returns:
            sensor_points: Radar points belonging to the given sample
                and sensor in the vehicle frame.
        """
        # Load sensor point cloud in sensor frame
        sensor_points, timestamps = nuscenes.RadarPointCloud.from_file_multisweep(nusc, sample, chan=sensor, ref_chan=sensor,
                                                                                  nsweeps=self.nsweeps, min_distance=1.0)

        sensor_points = np.vstack([sensor_points.points, timestamps])

        # Get calibration information
        sample_data = nusc.get('sample_data', sample['data'][sensor])
        calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

        # Get rotation matirx from sensor to vehicle frame
        rotation_matrix = Quaternion(calibrated_sensor['rotation']).rotation_matrix

        # Rotate point cloud
        sensor_points[:3, :] = np.dot(rotation_matrix, sensor_points[:3, :])

        # Translate point cloud
        sensor_points[:3, :] = np.add(sensor_points[:3, :], np.expand_dims(calibrated_sensor['translation'], axis=-1))

        # Rotate velocity vectors
        sensor_points[8:10, :] = np.dot(rotation_matrix[:2, :2], sensor_points[8:10, :])

        return sensor_points

    def get_bounding_boxes(self, nusc, sample: dict, sensor: str) -> List[nuscenes.Box]:
        """ Returns the bounding boxes belonging to a given sample and sensor.

        The bounding box labels are determined in accordance to the
        nuScenes detection challenge class definition.

        Arguments:
            nusc: nuScenes instance of the selected dataset version.
            sample: Single nuScenes sample instance.
            sensor: Single radar sensor instance.

        Returns:
            boxes: Bounding boxes belonging to the given sample and sensor
                in the global frame.
        """
        # Get bounding boxes
        boxes = nusc.get_boxes(sample['data'][sensor])

        # Determine box labels
        for box in boxes:
            box.label = self._get_box_label(box.name)

        return boxes

    def get_labels(self, nusc, sample: dict, sensor: str, points: np.ndarray) -> np.ndarray:
        """ Returns the class labels and bounding boxes for the given radar sensor data.

        Arguments:
            nusc: nuScenes instance of the selected dataset version.
            sample: Single nuScenes sample instance.
            sensor: Single radar sensor instance.
            points: Radar points of the given sensor and sample.

        Returns:
            sensor_labels: Class labels for the given radar points.
            boxes: Bounding boxes belonging to the given sample and sensor
                in the vehicle frame.
        """
        # Get ego pose
        sample_data = nusc.get('sample_data', sample['data'][sensor])
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])

        # Set point z-coordinate values (second dimension) to zero to
        # compensate interpolation errors (this method only works for
        # quasi 2d radar data).
        points[2, :] = 0.0

        # Initialize all points as invalid points
        number_of_radar_points = points.shape[1]
        sensor_labels = np.zeros(shape=[number_of_radar_points], dtype=int)

        # Get bounding boxes
        boxes = self.get_bounding_boxes(nusc, sample, sensor)

        # Filter bounding boxes
        boxes = self.filter_bounding_boxes(nusc, boxes)

        # Transform bounding boxes to vehicle frame
        for box in boxes:
            box.translate(np.multiply(ego_pose['translation'], -1))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)

        # Crop bounding boxes
        if self.dataset_config.crop_point_cloud:
            boxes = self.crop_bounding_boxes(boxes)

        # Determine point labels
        for box in boxes:
            # Determine if points are within the bounding box
            points_in_box = utils.extended_points_in_box(box=box, points=points[:3, :], wlh_factor=self.wlh_factor,
                                                         wlh_offset=self.wlh_offset, use_z=False)

            # Assign class label to all points within the bounding box.
            sensor_labels[points_in_box] = box.label

        return sensor_labels, boxes

    def create_graph_data(self, geometric_graph, labels, bounding_boxes, point_cloud) -> Data:
        """ Creates a geometric graph data element from its individual components.

        Arguments:
            geometric_graph: Graph data representing the model input.
            labels: Class labels for model training.
            bounding_boxes: Bounding boxes for model training.
            point_cloud: Point cloud corresponding to the graph data.

        Returns:
            data: PyTorch geometric graph data element.
        """

        # Concatenate class labels and bounding boxes
        targets = np.concatenate((labels, bounding_boxes), axis=1)

        # Transform to pytorch tensors
        x = torch.tensor(geometric_graph.X_feat, dtype=torch.float32)
        edge_index = torch.tensor(geometric_graph.E.T, dtype=torch.long)
        edge_attr = torch.tensor(geometric_graph.E_feat, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        X_pos = torch.tensor(point_cloud.X_cc, dtype=torch.float32)
        Vel = torch.tensor(point_cloud.V_cc_compensated, dtype=torch.float32)

        # Convert to pytorch geometric data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=y, pos=X_pos, vel=Vel)

        return data

    def process_single_sample(self, nusc, sample: dict, path_to_destination: str = None) -> None:
        """ Prepares a single nuScenes sample for model training or evaluation.

        This function creates both the model input graph as well as the target
        data for model training or evaluation from the given nuScenes sample.

        Arguments:
            nusc: nuScenes instance of the selected dataset version.
            sample: Single nuScenes sample instance.
            path_to_destination: Path to store the processed data.
        """
        # Initialize points (input) and labels (target)
        points = np.empty(shape=(nuscenes.RadarPointCloud.nbr_dims() + 1, 0))

        # Create point cloud from multiple sensors in vehicle frame
        for sensor in {'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'}:
            # Get sensor point cloud in vehicle frame
            sensor_points = self.get_sensor_points(nusc, sample, sensor)

            # Add sensor point cloud to sample point cloud
            points = np.append(points, sensor_points, axis=1)

        # Crop point cloud
        if self.dataset_config.crop_point_cloud:
            points = self.crop_point_cloud(points)

        # Get sensor point cloud lables and bounding boxes
        labels, boxes = self.get_labels(nusc, sample, sensor='LIDAR_TOP', points=points)

        # Convert points to point cloud
        point_cloud = conversion.convert_point_cloud(points, labels)

        # Construct geometric graph
        geometric_graph = conversion.build_geometric_graph(self.graph_config, point_cloud)

        # Convert bounding boxes
        bounding_boxes = conversion.convert_bounding_boxes(self.dataset_config, point_cloud, boxes,
                                                           wlh_factor=self.wlh_factor, wlh_offset=self.wlh_offset)

        # Create graph data
        graph = self.create_graph_data(geometric_graph, np.atleast_2d(labels).T, bounding_boxes, point_cloud)

        # Save graph
        save_name = f"{path_to_destination}/graph_{sample['timestamp']}_{sample['scene_token']}_{sample['token']}.pt"
        torch.save(graph, save_name)

    def process(self) -> None:
        """ Pre-processes and prepares the raw nuScenes radar data for model training."""
        dataset_split = NuScenesSplitConfiguration(version=self.dataset_config.version)
        self.dataset_split_config = dataset_split
        sequence_dict = self.dataset_split_config.sequence_dict

        # Read data from `raw_path`.
        for split_name, sequence_list in sequence_dict.items():

            print(f"\n************ Creating {split_name} dataset ************")

            path_to_nuScenes = f"{self.root}/raw"
            path_to_destination = f"{self.root}/processed/{split_name}"
            os.mkdir(path_to_destination)

            # create the config file with dataset information
            path_to_config = f"{self.root}/processed/config.json"
            create_data_set_description(
                self.graph_config, self.dataset_config, self.dataset_split_config, path_to_config)

            # get nuscenes instance
            nusc = nuscenes.NuScenes(version=self.dataset_config.version, dataroot=path_to_nuScenes, verbose=False)

            # create and save all graphs sequence by sequence
            for i, sequence in enumerate(sequence_list):
                print(f"----------- Processing Sequence {i+1}/{len(sequence_list)} -----------")

                # Get scene
                scene = nusc.get('scene', sequence)

                # Get first sample
                sample = nusc.get('sample', scene['first_sample_token'])

                # Process first sample
                self.process_single_sample(nusc, sample, path_to_destination)

                for _ in range(scene['nbr_samples'] - 1):
                    # Get sample
                    sample = nusc.get('sample', sample['next'])

                    # Process sample
                    self.process_single_sample(nusc, sample, path_to_destination)

    def len(self, split_name: str) -> int:
        pass

    def get(self, split_name: str, idx: int):
        pass


def create_data_set_description(graph_config: GraphConstructionConfiguration, dataset_config: NuScenesDatasetConfiguration,
                                split_config: NuScenesSplitConfiguration, path_to_destination: str) -> None:
    """
    Creates a json file with the description of the dataset and creation settings
    and saves it in to destination path
    """

    # create dictionaries that can then be written into Json file
    graph_config_dict = asdict(graph_config)
    dataset_config_dict = asdict(dataset_config)
    split_config_dict = asdict(split_config)

    json_dict = {"GRAPH_CONSTRUCTION_SETTINGS": graph_config_dict,
                 "DATASET_CONFIG": dataset_config_dict,
                 "DATASET_SPLIT_CONFIG": split_config_dict}

    # write JSON file
    with open(path_to_destination, 'w') as f:
        json.dump(json_dict, f, indent=4)
