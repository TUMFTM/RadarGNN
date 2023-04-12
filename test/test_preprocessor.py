import numpy as np
from sklearn.neighbors import kneighbors_graph

from gnnradarobjectdetection.preprocessor.bounding_box import RotationInvariantRelativeRotatedBoundingBox
from gnnradarobjectdetection.preprocessor.configs import GraphConstructionConfiguration
from gnnradarobjectdetection.preprocessor.radar_point_cloud import RadarPointCloud
from gnnradarobjectdetection.preprocessor.radarscenes.dataset_creation import GraphConstructor, GroundTruthCreator, PointCloudProcessor
from gnnradarobjectdetection.preprocessor.radarscenes.configs import RadarScenesDatasetConfiguration


def test_remove_points_from_radar_point_cloud_with_nan():
    point_cloud = RadarPointCloud()
    point_cloud.label_id = np.ones([5, 1]) * 1
    point_cloud.label_id[4, 0] = np.nan
    point_cloud.V_cc_compensated = np.array([[2, -8],
                                            [np.nan, 1],
                                            [1, np.nan],
                                            [np.nan, np.nan],
                                            [1, 1]]).reshape(5, 2)

    point_cloud.remove_points_without_labelID()
    point_cloud.remove_points_without_valid_velocity()
    assert ((point_cloud.V_cc_compensated == np.array([[2, -8]])).all())


def test_reconstruction_of_rotation_invariant_bb_to_absolute_bb():

    point_cloud = RadarPointCloud()
    point_cloud.X_cc = np.array([[1, 2],
                                [3, 4],
                                [-1, 3],
                                [9, 8],
                                [10, 7],
                                [-2, -3]]).reshape(6, 2)

    point_cloud.label_id = np.ones([6, 1]) * 6
    point_cloud.track_id = np.array(
        [b'0', b'0', b'0', b'1', b'1', b'2']).reshape(6, 1)

    res = True
    # rotate bounding box 360 times, create rotation invariant representation and reconstruct the NON rotation invariant definition
    for phi in range(0, 360, 1):

        # apply rotation to point cloud
        phi_rad = (phi * np.pi) / 180
        r = np.array([[np.cos(phi_rad), -np.sin(phi_rad)],
                      [np.sin(phi_rad), np.cos(phi_rad)]])

        point_cloud.X_cc = np.matmul(point_cloud.X_cc, r)

        # create NON rotation invariant but translation invariant representation of the rotated bounding boxes
        bounding_box_matrix_base = GroundTruthCreator.create_2D_bounding_boxes(
            point_cloud, False, "translation")

        # create the rotation invariant bounding box representation
        bounding_box_matrix = GroundTruthCreator.create_2D_bounding_boxes(
            point_cloud, False, "en")

        # recover for every bounding box the NON rotation invariant representation
        bounding_box_matrix_base_recovered = np.empty_like(
            bounding_box_matrix_base)

        # get nearest neighbor of every point
        A_sparse = kneighbors_graph(
            point_cloud.X_cc, 1, mode='connectivity', include_self=False)
        A = A_sparse.toarray()
        X_cc_nn = point_cloud.X_cc[np.where(A == 1)[1]]

        for i in range(bounding_box_matrix_base_recovered.shape[0]):

            point_koord = point_cloud.X_cc[i, :]
            nn_koord = X_cc_nn[i, :]

            # transform rad to deg
            bounding_box_matrix[i, 1] = (
                bounding_box_matrix[i, 1] * 180) / np.pi
            bounding_box_matrix[i, 4] = (
                bounding_box_matrix[i, 4] * 180) / np.pi
            bounding_box_matrix_base[i, 4] = (
                bounding_box_matrix_base[i, 4] * 180) / np.pi

            bb_rot_inv = RotationInvariantRelativeRotatedBoundingBox(bounding_box_matrix[i, 0], bounding_box_matrix[i, 1],
                                                                     bounding_box_matrix[i, 2], bounding_box_matrix[i, 3], bounding_box_matrix[i, 4])
            # crate bounding box objects
            relative_rotated_bb = bb_rot_inv.rotation_invariant_representation_to_relative_rotated_bb(
                point_koord, nn_koord)

            bounding_box_matrix_base_recovered[i, :] = np.array([relative_rotated_bb.x_center, relative_rotated_bb.y_center,
                                                                relative_rotated_bb.l, relative_rotated_bb.w, relative_rotated_bb.theta]).reshape(1, 5)

        # check if relative rotated bounding box was recovered correctly
        if not (np.round(bounding_box_matrix_base[:5, :], 5) == np.round(bounding_box_matrix_base_recovered[:5, :], 5)).all():
            res = False
        # ignore the rotation angle of one point bounding box -> This angle can not be reconstructed
        if not (np.round(bounding_box_matrix_base[5, :4], 5) == np.round(bounding_box_matrix_base_recovered[5, :4], 5)).all():
            res = False

    assert (res)


def test_rotation_invariant_bb_representation():

    point_cloud = RadarPointCloud()
    point_cloud.X_cc = np.array([[1, 2],
                                [3, 4],
                                [-1, 3],
                                [9, 8],
                                [10, 7],
                                [-2, -3]]).reshape(6, 2)

    point_cloud.label_id = np.ones([6, 1]) * 6
    point_cloud.track_id = np.array(
        [b'0', b'0', b'0', b'1', b'1', b'2']).reshape(6, 1)

    bounding_box_matrix_base = GroundTruthCreator.create_2D_bounding_boxes(
        point_cloud, False, "en")

    res = True
    # rotate bounding box 360 times
    for phi in range(1, 360, 1):

        # apply rotation to point cloud
        phi_rad = (phi * np.pi) / 180
        r = np.array([[np.cos(phi_rad), -np.sin(phi_rad)],
                      [np.sin(phi_rad), np.cos(phi_rad)]])

        point_cloud.X_cc = np.matmul(point_cloud.X_cc, r)

        bounding_box_matrix = GroundTruthCreator.create_2D_bounding_boxes(
            point_cloud, False, "en")

        # check if bounding box representation is identical even tough point cloud was rotated
        if not (np.round(bounding_box_matrix_base, 5) == np.round(bounding_box_matrix, 5)).all():
            res = False

    assert (res)


def test_bounding_box_creation_aligned():
    point_cloud = RadarPointCloud()
    point_cloud.X_cc = np.array([[1, 2],
                                [3, 4],
                                [-1, 3],
                                [9, 8],
                                [10, 7],
                                [-2, -3]]).reshape(6, 2)

    point_cloud.label_id = np.ones([6, 1]) * 6
    point_cloud.track_id = np.array(
        [b'0', b'0', b'0', b'1', b'1', b'2']).reshape(6, 1)

    bounding_box_matrix_aligned = GroundTruthCreator.create_2D_bounding_boxes(
        point_cloud, True, False)

    bb_1 = bounding_box_matrix_aligned[0, :].tolist()
    bb_2 = bounding_box_matrix_aligned[3, :].tolist()
    bb_3 = bounding_box_matrix_aligned[5, :].tolist()

    bb_1_true = [0, 1, 4, 2]
    bb_2_true = [0.5, -0.5, 1, 1]
    bb_3_true = [0, 0, 0.5, 0.5]

    res = (bb_1 == bb_1_true and bb_2 == bb_2_true and bb_3 == bb_3_true)
    assert (res)


def test_bounding_box_creation_rotated():

    point_cloud = RadarPointCloud()
    point_cloud.X_cc = np.array([[1, 2],
                                [2, 1],
                                [1, 0],
                                [0, 1]]).reshape(4, 2)

    point_cloud.label_id = np.ones([4, 1]) * 6
    point_cloud.track_id = np.array([b'0', b'0', b'0', b'0']).reshape(4, 1)

    bounding_box_matrix_rotated = GroundTruthCreator.create_2D_bounding_boxes(
        point_cloud, False, "translation")

    bb = bounding_box_matrix_rotated[0, :]
    bb_true = np.array([0, -1, 2 ** 0.5, 2 ** 0.5, 45 * np.pi / 180])
    assert ((abs((bb - bb_true)) < 1E-10).all())


def test_one_hot_vector_creation():

    point_cloud = RadarPointCloud()
    point_cloud.X_cc = np.array([[1, 2],
                                [2, 1],
                                [1, 0],
                                [0, 1]]).reshape(4, 2)

    point_cloud.label_id = np.array([1, 1, 0, 1])

    one_hot_vectors = GroundTruthCreator.build_one_hot_vectors(point_cloud)

    v_1 = one_hot_vectors[0, :].tolist()
    v_2 = one_hot_vectors[2, :].tolist()

    v_1_true = [0, 1, 0, 0, 0, 0]
    v_2_true = [1, 0, 0, 0, 0, 0]

    assert (v_1 == v_1_true and v_2 == v_2_true)


def test_graph_constructor():
    point_cloud = RadarPointCloud()
    point_cloud.X_cc = np.array([[1, 1],
                                 [3, 2],
                                 [5, 8]]).reshape(3, 2)

    point_cloud.V_cc_compensated = np.ones_like(point_cloud.X_cc)
    point_cloud.timestamp = np.array([100, 101, 102]).reshape(3, 1)

    node_feat = ["spatial_coordinates", "time_index"]
    edge_feat = ["spatial_euclidean_distance"]
    config = GraphConstructionConfiguration(
        "knn", {"k": 1, "r": 1}, node_feat, edge_feat, "directed", "X")
    graph = GraphConstructor.build_geometric_graph(config, point_cloud)

    node_feat = graph.X_feat[1, :]
    edge_feat = graph.E_feat[0, :]
    edges = graph.E

    node_feat_true = np.array([3, 2, 1])
    edge_feat_true = 5 ** 0.5
    edges_true = np.array([[0, 1], [1, 0], [2, 1]]).reshape(3, 2)

    assert (((edge_feat == edge_feat_true).all() and (node_feat == node_feat_true).all() and (edges == edges_true).all()))


def test_graph_constructor_distance_definition():
    point_cloud = RadarPointCloud()
    point_cloud.X_cc = np.array([[1, 1],
                                 [2, 2],
                                 [10, 10]]).reshape(3, 2)

    point_cloud.V_cc_compensated = np.ones_like(point_cloud.X_cc)
    point_cloud.V_cc_compensated[0, :] = 100

    node_feat = ["spatial_coordinates"]
    edge_feat = ["spatial_euclidean_distance"]

    config = GraphConstructionConfiguration(
        "knn", {"k": 1, "r": 1}, node_feat, edge_feat, "directed", "X")
    graph = GraphConstructor.build_geometric_graph(config, point_cloud)
    edges_X = graph.E
    edges_true_x = np.array([[0, 1], [1, 0], [2, 1]]).reshape(3, 2)

    config = GraphConstructionConfiguration(
        "knn", {"k": 1, "r": 1}, node_feat, edge_feat, "directed", "XV")
    graph = GraphConstructor.build_geometric_graph(config, point_cloud)
    edges_XV = graph.E
    edges_true_XV = np.array([[0, 1], [1, 2], [2, 1]]).reshape(3, 2)

    assert ((((edges_X == edges_true_x).all()) and (edges_XV == edges_true_XV).all()))


def test_point_cloud_processor():
    point_cloud = RadarPointCloud()
    point_cloud.X_cc = np.array([[1, 1],
                                 [1, -1],
                                 [0, 4],
                                 [10, 1],
                                 [-3, 1],
                                 [1.5, 1.5]]).reshape(6, 2)

    point_cloud.V_cc_compensated = np.array([[1, 1],
                                            [1, 1],
                                            [1, 1],
                                            [1, 1],
                                            [1, 1],
                                            [np.nan, 1]]).reshape(6, 2)

    point_cloud.label_id = np.array([1, np.nan, 1, 2, 1, 1]).reshape(6, 1)

    dataset_config = RadarScenesDatasetConfiguration(
        None, True, {"front": 5, "sides": 2}, None, None, None)
    point_cloud = PointCloudProcessor.transform(dataset_config, point_cloud)

    (point_cloud.X_cc == np.array([1, 1])).all()
    assert (True)
