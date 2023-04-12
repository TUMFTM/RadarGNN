import os
import sys
import glob
import json
from dataclasses import asdict
from shutil import rmtree
import random
from math import atan2

import ray
import torch
from torch_geometric.data import Data, Dataset
from sklearn.neighbors import kneighbors_graph
import numpy as np
import matplotlib.pyplot as plt
from radar_scenes.sequence import Sequence

from radargnn.preprocessor.configs import GraphConstructionConfiguration
from radargnn.preprocessor.radar_point_cloud import RadarPointCloud

from radargnn.preprocessor.radarscenes.configs import RadarScenesDatasetConfiguration, RadarScenesSplitConfiguration
from radargnn.preprocessor.radarscenes.scene_collection import concatenate_subsequent_scenes
from radargnn.graph_constructor.graph import GeometricGraph
from radargnn.utils.math import minimum_bounding_rectangle_without_rotation, minimum_bounding_rectangle_with_rotation_alternative
from radargnn.preprocessor.bounding_box import BoundingBox, RelativeAlignedBoundingBox, RelativeRotatedBoundingBox, RotationInvariantRelativeRotatedBoundingBox, AbsoluteRotatedBoundingBox


class RadarScenesGraphDataset(Dataset):
    def __init__(self, root: str, graph_config: GraphConstructionConfiguration,
                 dataset_config: RadarScenesDatasetConfiguration, transform=None,
                 pre_transform=None, pre_filter=None):

        # to be defined before calling super
        self.graph_config = graph_config
        self.dataset_config = dataset_config

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

        print("Download not possible and no RadarScenes dataset in given root path ")
        print("Should the following automatically created folder be deleted again ?\n")
        print(self.root[:-21])

        print("\nThe folder contains the following files: \n")

        files = glob.glob(f'{self.root}/*')
        for f in files:
            print(f"- {f}")

        answer = input("\ny/n: ")

        if answer == "y":
            rmtree(self.root[:-21])
            print("Folder deleted")
        else:
            pass

        sys.exit(
            "Download RadarScenes dataset manually and store it in data/raw folder")

    def process(self):
        """ Processes the dataset.

        Creates three graph datasets (train, test, validation) based on the RadarScenes data and saves them.
        """
        # create the dataset split configuration
        sequence_file = f"{self.root}/raw/data/sequences.json"
        dataset_split = RadarScenesSplitConfiguration(sequence_file, standard_split=True)
        self.dataset_split_config = dataset_split
        sequence_dict = self.dataset_split_config.sequence_dict

        # Read data from `raw_path`.
        for split_name, sequence_list in sequence_dict.items():

            print(f"\n************ Creating {split_name} dataset ************")

            path_to_RadarScenes = f"{self.root}/raw"
            path_to_destination = f"{self.root}/processed/{split_name}"
            os.mkdir(path_to_destination)

            # get random scenes if only small subset is created
            if self.dataset_config.create_small_subset and split_name == 'train':
                # get new sequences for training / testing / validation sequence list
                num_sequences = self.dataset_config.subset_settings["num_sequences"]
                random.shuffle(sequence_list)
                sequence_list = sequence_list[:num_sequences]

                # add new sequences to dataset config
                self.dataset_split_config.sequence_dict[split_name] = sequence_list

            # create the config file with dataset information
            path_to_config = f"{self.root}/processed/config.json"
            create_data_set_description(
                self.graph_config, self.dataset_config, self.dataset_split_config, path_to_config)

            if self.dataset_config.parallelize:
                # process multiple sequences parallel (all train/test/validate sequences)
                # ATTENTION: Consumes a lot of RAM -> Can be further optimized
                ray.init()
                num_worker = int(ray.available_resources()["CPU"])
                print(f"Using {num_worker} workers for parallel dataset processing")

                graph_data_list = create_data_set_from_radar_scenes(
                    self.graph_config, self.dataset_config, sequence_list, path_to_RadarScenes)

                # save the graphs
                for graph_idx, graph in enumerate(graph_data_list):
                    save_name = f'{path_to_destination}/graph_{graph_idx}.pt'
                    torch.save(graph, save_name)

                # give free storage by removing graph data list
                graph_data_list = None
                ray.shutdown()

            else:
                # create and save all graphs sequence by sequence
                graph_idx = 0
                for i, sequence in enumerate(sequence_list):
                    print(f"----------- Processing Sequence {i+1}/{len(sequence_list)} -----------")

                    # create all graphs of one sequence
                    graph_data_list = create_graph_data_from_one_radar_scenes_sequence(
                        self.graph_config, self.dataset_config, sequence, path_to_RadarScenes)

                    # save the graphs
                    for graph in graph_data_list:
                        save_name = f'{path_to_destination}/graph_{graph_idx}.pt'
                        torch.save(graph, save_name)
                        graph_idx += 1

                    # give free storage by removing graph data list
                    graph_data_list = None

    def len(self, split_name):
        path_to_split = f"{self.root}/processed/{split_name}"
        list = os.listdir(path_to_split)  # dir is your directory path
        return len(list)

    def get(self, split_name, idx):
        path_to_graph = f"{self.root}/processed/{split_name}/graph_{idx}.pt"
        data = torch.load(path_to_graph)
        return data


class PointCloudProcessor():

    @staticmethod
    def transform(dataset_config: RadarScenesDatasetConfiguration,
                  point_cloud: RadarPointCloud) -> RadarPointCloud:
        """ Preprocesses the point cloud.

        Crops the point cloud and removes invalid points.

        Args:
            dataset_config: The dataset configuration.
            point_cloud: A radar point cloud.

        Returns:
            point_cloud: The radar point cloud with removed points.
        """

        if dataset_config.crop_point_cloud:
            front = dataset_config.crop_settings.get("front")
            sides = dataset_config.crop_settings.get("sides")
            point_cloud.remove_points_out_of_range(front, sides)

        point_cloud.remove_points_without_labelID()
        point_cloud.remove_points_without_valid_velocity()

        return point_cloud


class GraphConstructor():

    @staticmethod
    def build_geometric_graph(config: GraphConstructionConfiguration,
                              point_cloud: RadarPointCloud) -> GeometricGraph:
        """ Builds a graph from a point cloud based on the configuration.

        Args:
            config: The graph construction configuration.
            point_cloud: A radar point cloud.

        Returns:
            graph: The graph constructed from the point cloud.
        """

        # build the graph
        if config.distance_definition == "X":
            distance_basis = point_cloud.X_cc
        elif config.distance_definition == "XV":
            distance_basis = np.concatenate((point_cloud.X_cc, point_cloud.V_cc_compensated), axis=1)

        graph = GeometricGraph()
        graph.X = point_cloud.X_cc
        graph.V = point_cloud.V_cc_compensated
        graph.F = {"rcs": point_cloud.rcs}

        # add time-index as node feature
        if "time_index" in config.node_features:
            timestamps = np.unique(point_cloud.timestamp)
            t_idx = np.zeros_like(point_cloud.timestamp)

            for i, _ in enumerate(timestamps):
                idx = np.where(point_cloud.timestamp == timestamps[i])[0]
                t_idx[idx] = int(i)

            # add time index to invariant graph features
            graph.add_invariant_feature("time_index", t_idx)

        graph.build(distance_basis, config.graph_construction_algorithm, k=config.k, r=config.r)
        graph.extract_node_pair_features(config.edge_features, config.edge_mode)
        graph.extract_single_node_features(config.node_features)

        return graph


class GroundTruthCreator():

    @staticmethod
    def get_class_indices(point_cloud: RadarPointCloud) -> np.ndarray:
        return point_cloud.label_id

    @staticmethod
    def build_one_hot_vectors(point_cloud: RadarPointCloud) -> np.ndarray:
        num_classes = 6
        target = np.zeros([point_cloud.label_id.shape[0], num_classes])
        for i, label in enumerate(point_cloud.label_id):
            target[i, int(label)] = 1

        return target

    @classmethod
    def create_2D_bounding_boxes(cls, point_cloud: RadarPointCloud, aligned: bool, bb_invariance: str) -> np.ndarray:
        """ Creates 2D ground truth bounding boxes for a point cloud.

        Args:
            point_cloud: The point cloud.
            aligned: Bool to decide whether boxes should be aligned or rotated.
            bb_invariance: Bounding box representation / its level of invariance (For rotated boxes).

        Returns:
            bounding_box_matrix: Matrix with each row containing the created bounding box of the corresponding node's object.

        TODO:
            - Add possibility to normalize bb w.r.t mean object size (as in Point-GNN)
            - Maybe predict theta as probability over 8 possible bins and not regress exact angle (as in Radar Point-GNN)
        """

        if aligned:
            bounding_box_matrix = cls.__build_bounding_boxes_without_rotation(point_cloud)
        else:
            bounding_box_matrix = cls.__build_bounding_boxes_with_rotation(point_cloud, bb_invariance)

        return bounding_box_matrix

    @staticmethod
    def __build_bounding_boxes_with_rotation(point_cloud: RadarPointCloud, bb_invariance: str) -> np.ndarray:
        """ Creates rotated ground truth bounding boxes.

        Note:
            Non invariant bounding boxes are of the format (bb_invariance = "none"):
            [x, y, length, width, theta]
                - x: absolute x position of bb center
                - y: absolute y position of bb center
                - length: size of longer bb edge
                - width: size of shorter bb edge
                - theta: relative rotation of the long bb axis w.r.t the x-axis (so far) IN RAD

            Translation invariant bounding boxes are of the format (bb_invariance = "translation"):
            [x_center, y_center, length, width, theta]
                - x_center: relative x position of bb center w.r.t the point
                - y_center: relative y position of bb center w.r.t the point
                - length: size of longer bb edge
                - width: size of shorter bb edge
                - theta: relative rotation of the long bb axis w.r.t the x-axis (so far) IN RAD

            Rotation invariant bounding boxes are of the format (bb_invariance = "en"):
            [d, theta_v_p_nn_v_p_c, l, w, theta_v_p_nn_v_dir]
                - d: distance of the point to the bb center
                - theta_v_p_nn_v_p_c: clockwise angle from v_p_nn to v_p_c IN RAD
                    - v_p_nn: vector from point to its nearest neighbor
                    - v_p_c: vect from point to the bounding box center
                - l: size of longer bb edge
                - w: size of shorter bb edge
                - theta_v_p_nn_v_dir: clockwise angle from v_p_nn to v_dir IN RAD
                    - v_p_nn: vector from point to its nearest neighbor
                    - v_dir: vector in direction of the longer bb edge
        """

        # get IDs of all object in that point cloud
        object_track_ids = np.unique(point_cloud.track_id)
        background_id = np.where(object_track_ids == b'')[0]
        object_track_ids = np.delete(object_track_ids, background_id)

        # iterate through all objects and create their bounding box
        bounding_box_matrix = np.zeros([point_cloud.X_cc.shape[0], 5])
        bounding_box_matrix[:] = np.nan

        if bb_invariance == "en":
            # get nearest neighbor of each point in the point cloud
            A_sparse = kneighbors_graph(point_cloud.X_cc, 1, mode='connectivity', include_self=False)
            A_full = A_sparse.toarray()
            X_cc_nn = point_cloud.X_cc[np.where(A_full == 1)[1]]

        for object_id in object_track_ids:
            object_idx = np.where(point_cloud.track_id == object_id)[0]

            if object_idx.shape[0] == 1:
                # object consists of a single radar target only

                # get relative position of bb center w.r.t point
                if bb_invariance == "none":
                    # get coordinates of the point
                    x = point_cloud.X_cc[object_idx, 0][0]
                    y = point_cloud.X_cc[object_idx, 1][0]

                else:
                    x = 0
                    y = 0

                # length and width
                l = 0.5
                w = 0.5

                # theta must be fixed to be rotation invariant
                theta_x = 0
                bb_array = np.array([x, y, l, w, theta_x]).reshape(1, 5)
                bounding_box_matrix[object_idx, :] = bb_array

            elif object_idx.shape[0] == 2:
                # get coordinates of both points
                point_coords = np.array([point_cloud.X_cc[i, :] for i in object_idx])
                p1 = point_coords[0, :]
                p2 = point_coords[1, :]

                # get center of the bounding box
                c = ((p1 + p2) / 2).reshape(1, 2)

                # long bb side is connection vector between the two points
                v_l = (p2 - p1).reshape(2, 1)

                # get angle between longe side direction vector and x-axis in range [-180, 180]
                v_l_norm = v_l / np.linalg.norm(v_l)
                theta_x = atan2(v_l_norm[1, 0], v_l_norm[0, 0]) * 180 / np.pi

                # transform to tange between 0-180 -> e.g. -45° and 135° is same bounding box
                if theta_x < 0:
                    theta_x = 180 + theta_x

                # distance between the points = length, width = 0.5
                l = np.linalg.norm(v_l)
                w = 0.5

                # iterate through all points within the box
                for idx in object_idx:

                    # get points coordinates
                    point_coord = point_cloud.X_cc[idx, :]
                    x = point_coord[0]
                    y = point_coord[1]

                    # get relative position of bb center w.r.t point
                    relative_position = c - np.array([x, y]).reshape(1, 2)
                    x_rel = relative_position[0, 0]
                    y_rel = relative_position[0, 1]

                    # get relative rotated bounding box
                    if bb_invariance == "en":
                        relative_bb = RelativeRotatedBoundingBox(x_rel, y_rel, l, w, theta_x)
                        nn_coord = X_cc_nn[idx, :]
                        relative_bb_rot_inv = relative_bb.relative_rotated_bb_to_rotation_invariant_representation(point_coord, nn_coord)
                        bb_array = np.array([relative_bb_rot_inv.d, relative_bb_rot_inv.theta_v_p_nn_v_p_c,
                                             relative_bb_rot_inv.l, relative_bb_rot_inv.w, relative_bb_rot_inv.theta_v_p_nn_v_dir]).reshape(1, 5)

                    elif bb_invariance == "none":
                        bb_array = np.array([c[0, 0], c[0, 1], l, w, theta_x]).reshape(1, 5)

                    elif bb_invariance == "translation":
                        bb_array = np.array([x_rel, y_rel, l, w, theta_x]).reshape(1, 5)

                    else:
                        raise ValueError("Wrong invariance for bounding box selection")

                    # convert from degree to rad
                    # TODO add option in configuration whether using degree or radians for the angle representations
                    if bb_invariance == "en":
                        bb_array[0, 1] = (bb_array[0, 1] * np.pi) / 180
                        bb_array[0, 4] = (bb_array[0, 4] * np.pi) / 180
                    else:
                        bb_array[0, 4] = (bb_array[0, 4] * np.pi) / 180

                    bounding_box_matrix[idx, :] = bb_array

            else:
                # calculate bounding box for the object

                # calculate enclosing rectangle
                point_coords = np.array([point_cloud.X_cc[i, :] for i in object_idx])

                bbox = minimum_bounding_rectangle_with_rotation_alternative(point_coords)

                bb = BoundingBox(bbox, False)

                # iterate through all points within the box
                for idx in object_idx:

                    # get relative rotated bounding box
                    point_coord = point_cloud.X_cc[idx, :]
                    x = point_coord[0]
                    y = point_coord[1]
                    relative_bb = bb.get_relative_bounding_box(x, y)

                    if bb_invariance == "en":
                        nn_coord = X_cc_nn[idx, :]
                        relative_bb_rot_inv = relative_bb.relative_rotated_bb_to_rotation_invariant_representation(point_coord, nn_coord)
                        bb_array = np.array([relative_bb_rot_inv.d, relative_bb_rot_inv.theta_v_p_nn_v_p_c,
                                             relative_bb_rot_inv.l, relative_bb_rot_inv.w, relative_bb_rot_inv.theta_v_p_nn_v_dir]).reshape(1, 5)

                    elif bb_invariance == "none":
                        x_c = point_coord[0] + relative_bb.x_center
                        y_c = point_coord[1] + relative_bb.y_center
                        bb_array = np.array([x_c, y_c, relative_bb.l, relative_bb.w, relative_bb.theta]).reshape(1, 5)

                    elif bb_invariance == "translation":
                        # store bounding box in matrix
                        bb_array = np.array([relative_bb.x_center, relative_bb.y_center,
                                            relative_bb.l, relative_bb.w, relative_bb.theta]).reshape(1, 5)
                    else:
                        raise ValueError("Wrong invariance for bounding box selection")

                    # convert from degree to rad
                    # TODO add option in configuration whether using degree or radians for the angle representations
                    if bb_invariance == "en":
                        bb_array[0, 1] = (bb_array[0, 1] * np.pi) / 180
                        bb_array[0, 4] = (bb_array[0, 4] * np.pi) / 180
                    else:
                        bb_array[0, 4] = (bb_array[0, 4] * np.pi) / 180

                    bounding_box_matrix[idx, :] = bb_array

        return bounding_box_matrix

    @staticmethod
    def __build_bounding_boxes_without_rotation(point_cloud: RadarPointCloud) -> np.ndarray:
        """ Creates axis-aligned ground truth bounding boxes.

        Note:
            Bounding boxes are of the format:
            [x, y, dx, dy]
                - x: relative x position of bb center w.r.t the point
                - y: relative y position of bb center w.r.t the point
                - dx: size in x direction
                - dy: size in y direction
        """

        # get IDs of all object in that point cloud
        object_track_ids = np.unique(point_cloud.track_id)
        background_id = np.where(object_track_ids == b'')[0]
        object_track_ids = np.delete(object_track_ids, background_id)

        # iterate through all objects and create their bounding box
        bounding_box_matrix = np.zeros([point_cloud.X_cc.shape[0], 4])
        bounding_box_matrix[:] = np.nan

        for object_id in object_track_ids:
            object_idx = np.where(point_cloud.track_id == object_id)[0]

            if object_idx.shape[0] == 1:
                # object consists of a single radar target only

                # get relative position of bb center w.r.t point
                x_rel = 0
                y_rel = 0

                # length and width
                dx = 0.5
                dy = 0.5

                # save bbox encoding in matrix
                bb = np.array([x_rel, y_rel, dx, dy]).reshape(1, 4)
                bounding_box_matrix[object_idx, :] = bb

            else:
                # calculate bounding box for the object
                # calculate enclosing rectangle
                point_coords = np.array([point_cloud.X_cc[i, :] for i in object_idx])
                bbox = minimum_bounding_rectangle_without_rotation(point_coords)

                bb = BoundingBox(bbox, True)

                # iterate through all points within the box
                for idx in object_idx:

                    # get relative rotated bounding box
                    point_coord = point_cloud.X_cc[idx, :]
                    x = point_coord[0]
                    y = point_coord[1]
                    relative_bb = bb.get_relative_bounding_box(x, y)

                    # store bounding box in matrix
                    bb_array = np.array([relative_bb.x_center, relative_bb.y_center,
                                         relative_bb.dx, relative_bb.dy]).reshape(1, 4)

                    bounding_box_matrix[idx, :] = bb_array

        return bounding_box_matrix

    def show_point_cloud_with_bounding_boxes(point_cloud: RadarPointCloud, bounding_box_matrix: np.ndarray, bb_invariance: str):
        """ Visualizes the point cloud with its created ground truth bounding boxes.
        """
        # draw point cloud
        _, ax = point_cloud.show(show_velocity_vector=False)
        plt.axis("equal")

        # get nearest neighbor of each point
        if bb_invariance == "en":
            A_sparse = kneighbors_graph(point_cloud.X_cc, 1, mode='connectivity', include_self=False)
            A_full = A_sparse.toarray()
            X_cc_nn = point_cloud.X_cc[np.where(A_full == 1)[1]]

        for i in range(bounding_box_matrix.shape[0]):
            if not np.isnan(bounding_box_matrix[i, 0]):

                # crate bounding box objects
                if bounding_box_matrix.shape[1] == 5:
                    bb = bounding_box_matrix[i, :].reshape(1, 5)
                    if bb_invariance == "en":
                        # extract bounding box information
                        point_koord = point_cloud.X_cc[i, :]
                        nn_koord = X_cc_nn[i, :]
                        bb = RotationInvariantRelativeRotatedBoundingBox(bb[0, 0], bb[0, 1], bb[0, 2], bb[0, 3], bb[0, 4])
                        bb_abs = bb.get_absolute_bounding_box(point_koord, nn_koord)

                    elif bb_invariance == "none":
                        bb = AbsoluteRotatedBoundingBox(bb[0, 0], bb[0, 1], bb[0, 2], bb[0, 3], bb[0, 4])
                        bb_abs = bb.get_absolute_bounding_box()

                    elif bb_invariance == "translation":
                        # extract bounding box information
                        bb = RelativeRotatedBoundingBox(bb[0, 0], bb[0, 1], bb[0, 2], bb[0, 3], bb[0, 4])

                        # get corresponding point coordinates
                        point_koord = point_cloud.X_cc[i, :].reshape(1, 2)
                        x = point_koord[0, 0]
                        y = point_koord[0, 1]

                        # get absolute bounding box defined by its corners and plot
                        bb_abs = bb.get_absolute_bounding_box(x, y)

                    else:
                        raise ValueError("Wrong invariance for bounding box selection")
                else:
                    # extract bounding box information
                    bb = bounding_box_matrix[i, :].reshape(1, 4)
                    x_rel = bb[0, 0]
                    y_rel = bb[0, 1]
                    dx = bb[0, 2]
                    dy = bb[0, 3]
                    bb = RelativeAlignedBoundingBox(x_rel, y_rel, dx, dy)

                    # get corresponding point coordinates
                    point_koord = point_cloud.X_cc[i, :].reshape(1, 2)
                    x = point_koord[0, 0]
                    y = point_koord[0, 1]

                    # get absolute bounding box defined by its corners and plot
                    bb_abs = bb.get_absolute_bounding_box(x, y)

                bb_abs.plot_on_axis(ax)


def create_data_set_from_radar_scenes(graph_config: GraphConstructionConfiguration, dataset_config: RadarScenesDatasetConfiguration,
                                      sequence_list: list, path_to_RadarScenes: str, path_to_destination: str = None) -> list:

    """ Transforms the data of multiple RadarScenes sequences into graphs.

    Args:
        graph_config: Graph construction configuration.
        dataset_config: Dataset configuration.
        sequence_list: List of sequence numbers to process.
        path_to_RadarScenes: Path to the RadarScenes dataset.
        path_to_destination: Path to store the created graphs.

    Returns:
        graph_data_list: List with all graphs and its corresponding ground truth data created from the sequences.
    """

    # Old Code (without parallelization)
    # graph_data_list = []
    # for i, sequence in enumerate(sequence_list):
    #    print(f"----------- Processing Sequence {i+1}/{len(sequence_list)} -----------")
    #
    #    graph_data_list_sequence = create_graph_data_from_one_radar_scenes_sequence(
    #        graph_config, dataset_config, sequence, path_to_RadarScenes, path_to_destination)
    #    graph_data_list.extend(graph_data_list_sequence)

    # parallel computing with ray
    graph_data_list_of_lists = ray.get([create_graph_data_from_one_radar_scenes_sequence_parallel.remote(
        graph_config, dataset_config, sequence, path_to_RadarScenes, path_to_destination, i) for i, sequence in enumerate(sequence_list)])

    # reduce the list of lists to one big list
    graph_data_list = []
    for list in graph_data_list_of_lists:
        graph_data_list.extend(list)

    return graph_data_list


@ray.remote
def create_graph_data_from_one_radar_scenes_sequence_parallel(graph_config: GraphConstructionConfiguration, dataset_config: RadarScenesDatasetConfiguration,
                                                              sequence_name: str, path_to_RadarScenes: str, path_to_destination: str = None, idx: int = 0) -> list:
    """ Transforms the data of one RadarScenes sequences into graphs.

    Supposed to be used in parallel on multiple CPUs to accelerate graph dataset creation.
    Accumulates the points from multiple scenes of the sequence.
    Preprocesses the point cloud frames.
    Builds the graphs from the created point cloud frames.
    Creates ground truth bounding boxes.
    Stores all information as a list of graph data objects.

    Args:
        graph_config: Graph construction configuration.
        dataset_config: Dataset configuration.
        sequence_name: Name of the sequence to process
        path_to_RadarScenes: Path to the RadarScenes dataset.

    Returns:
        graph_data_list: List with all graphs and its corresponding ground truth data created from one sequence.
    """

    print(f"----------- Processing Sequence {idx+1} -----------")
    point_clouds = create_point_cloud_frames(
        path_to_RadarScenes, sequence_name, dataset_config)
    graph_data_list = []

    for point_cloud in point_clouds:
        if point_cloud.X_cc.shape[0] > 1:
            graph = GraphConstructor.build_geometric_graph(graph_config, point_cloud)
            # target = GroundTruthCreator.build_one_hot_vectors(point_cloud)
            target = GroundTruthCreator.get_class_indices(point_cloud)
            bounding_box = GroundTruthCreator.create_2D_bounding_boxes(
                point_cloud, dataset_config.bounding_boxes_aligned, dataset_config.bb_invariance)

            graph_data = create_graph_data(graph, target, bounding_box, point_cloud)
            graph_data_list.append(graph_data)

    print(f">>> finished graph data creation for sequence {idx+1}")

    return graph_data_list


def create_graph_data_from_one_radar_scenes_sequence(graph_config: GraphConstructionConfiguration, dataset_config: RadarScenesDatasetConfiguration,
                                                     sequence_name: str, path_to_RadarScenes: str, path_to_destination: str = None) -> list:
    """ Transforms the data of one RadarScenes sequences into graphs.

    Accumulates the points from multiple scenes of the sequence.
    Preprocesses the point cloud frames.
    Builds the graphs from the created point cloud frames.
    Creates ground truth bounding boxes.
    Stores all information as a list of graph data objects.

    Args:
        graph_config: Graph construction configuration.
        dataset_config: Dataset configuration.
        sequence_name: Name of the sequence to process
        path_to_RadarScenes: Path to the RadarScenes dataset.

    Returns:
        graph_data_list: List with all graphs and its corresponding ground truth data created from one sequence.
    """

    print(">>> creating point cloud frames")
    point_clouds = create_point_cloud_frames(
        path_to_RadarScenes, sequence_name, dataset_config)
    graph_data_list = []

    print(f"finished with {len(point_clouds)} point clouds")

    print(">>> creating graph data objects from point clouds")

    for i, point_cloud in enumerate(point_clouds):

        if point_cloud.X_cc.shape[0] > 1:
            graph = GraphConstructor.build_geometric_graph(graph_config, point_cloud)
            # target = GroundTruthCreator.build_one_hot_vectors(point_cloud)
            target = GroundTruthCreator.get_class_indices(point_cloud)
            bounding_box = GroundTruthCreator.create_2D_bounding_boxes(
                point_cloud, dataset_config.bounding_boxes_aligned, dataset_config.bb_invariance)

            graph_data = create_graph_data(graph, target, bounding_box, point_cloud)
            graph_data_list.append(graph_data)

            print(f"graphs created: {i+1}/{len(point_clouds)}")
        else:
            print("graph left out due to emtpy point cloud")

    print("finished graph creation for that sequence")
    return graph_data_list


def create_point_cloud_frames(path_to_RadarScenes: str, sequence_name: str,
                              dataset_config: RadarScenesDatasetConfiguration,) -> list:
    """  Creates point cloud frames from one sequence.

    Accumulates multiple scenes of the sequence.
    Transforms the accumulated data of the scenes into point cloud frames.

    Args:
        path_to_RadarScenes: Path to the RadarScenes dataset.
        sequence_name: Name of the sequence to process.
        dataset_config: Dataset configuration.

    Returns:
        point_clouds: List of point clouds created from the one sequence.

    """

    path_to_sequence = f"{path_to_RadarScenes}/data/{sequence_name}/scenes.json"
    sequence = Sequence.from_json(path_to_sequence)
    timstamps = sequence.timestamps

    # get the first and last timestamp of the sequence
    start_time_stamp = timstamps[np.where(timstamps == np.min(timstamps))[0][0]]
    end_time_stamp_final = timstamps[np.where(timstamps == np.max(timstamps))[0][0]]
    end_time_stamp = 0

    # created point clouds based on summarizing scenes
    point_clouds = []

    # iterate through point clouds frames / collections of scenes within a defined time span
    count = 0
    while end_time_stamp != end_time_stamp_final:
        count += 1
        scene_collection = concatenate_subsequent_scenes(
            sequence, start_time_stamp, dataset_config.time_per_point_cloud_frame)

        last_scene = scene_collection.scenes[-1]

        last_scene_timestamp = last_scene.timestamp

        # required to extract all required information of the point cloud
        scene_collection.process(use_reduced_classes=True)
        point_cloud = scene_collection.point_cloud

        # process point cloud
        point_cloud = PointCloudProcessor.transform(dataset_config, point_cloud)
        point_clouds.append(point_cloud)

        # scene_collection.show()
        # plt.title(f"{sequence_name}, starting: {first_scene_timestamp}")

        start_time_stamp = last_scene_timestamp
        end_time_stamp = last_scene_timestamp

        # Get the first X point clouds per Sequence if a subset is created
        # only create defined number of point clouds
        # if dataset_config.create_small_subset:
        #    if count == dataset_config.subset_settings.get("num_clouds_per_sequence"):
        #        break

    # create all point clouds of the sequence and in the end use every n-th point cloud for the subset
    # equally discretize the time dimension in every sequence
    if dataset_config.create_small_subset and 'num_clouds_per_sequence' in dataset_config.subset_settings:
        num_pc_available = len(point_clouds)
        num_pc_allowed = dataset_config.subset_settings.get("num_clouds_per_sequence")
        point_clouds = [point_clouds[i] for i in np.floor(np.linspace(0, num_pc_available - 1, num_pc_allowed)).astype(int)]

    return point_clouds


def create_graph_data(graph: GeometricGraph, target: np.ndarray,
                      bounding_box: np.ndarray, point_cloud: RadarPointCloud) -> Data:
    """ Stores all required information of one graph sample into a graph data object.

    Args:
        graph: The graph constructed from the point cloud.
        target: The ground truth class labels of all points/nodes.
        bounding_box: The ground truth bounding boxes of all points/nodes.
        point_cloud: The point cloud itself.

    Returns:
        data: Graph data object containing all information.
    """

    # merge class labels and bounding boxes
    merged = np.concatenate((target, bounding_box), axis=1)

    # transform to tensors and store in pytorch geometric data object
    X_feat = torch.tensor(graph.X_feat, dtype=torch.float32)
    edge_index = torch.tensor(graph.E.T, dtype=torch.long)
    edge_attr = torch.tensor(graph.E_feat, dtype=torch.float32)
    y_merged = torch.tensor(merged, dtype=torch.float32)
    X_pos = torch.tensor(point_cloud.X_cc, dtype=torch.float32)
    Vel = torch.tensor(point_cloud.V_cc_compensated, dtype=torch.float32)

    data = Data(x=X_feat, edge_index=edge_index,
                edge_attr=edge_attr, y=y_merged, pos=X_pos, vel=Vel)

    return data


def create_data_set_description(graph_config: GraphConstructionConfiguration, dataset_config: RadarScenesDatasetConfiguration,
                                split_config: RadarScenesSplitConfiguration, path_to_destination: str) -> None:
    """ Creates a json file with the description of the dataset- and graph construction settings and saves it.
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
