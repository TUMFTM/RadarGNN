import copy
import os
from typing import Dict, List, Tuple

import numpy as np

from nuscenes import nuscenes
from pyquaternion import Quaternion


def transform_bounding_box(bounding_box: np.ndarray, nusc: nuscenes.NuScenes, sample_token: str) -> np.ndarray:
    """ Transforms the bounding box from the vehicle to the global frame.

    Arguments:
        bounding_box: Bounding box in the vehicle frame.
        nusc: nuScenes instance.
        sample_token: Token of the corresponding nuScenes sample.

    Returns:
        bounding_box: Bounding box in the global frame.
    """
    # Get ego pose
    sample = nusc.get('sample', sample_token)
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])

    # Get transformation values
    rot_matrix = Quaternion(ego_pose['rotation']).rotation_matrix
    trans = np.array(ego_pose['translation'])
    yaw, _, _ = Quaternion(ego_pose['rotation']).yaw_pitch_roll

    # Rotate bounding box
    bounding_box[:3] = np.squeeze(np.dot(rot_matrix, bounding_box[:3, None]))

    # Translate bounding box
    bounding_box[:3] = np.add(bounding_box[:3], trans)

    # Update bounding box angle
    bounding_box[-1] = np.deg2rad(bounding_box[-1]) + yaw

    return bounding_box


def get_3d_bounding_box(bounding_box: np.ndarray, size: Tuple[float, float, float]) -> np.ndarray:
    """

    Arguments:
        bounding_box: Bounding box in the format
            [center_x, center_y, length, width, theta_x]
        size: Bounding box size given as (width, length, height)

    Returns:
        bounding_box: Bounding box in the format
            [center_x, center_y, center_z, width, length, height, theta_x]
    """
    bounding_box = np.array([
        bounding_box[0],
        bounding_box[1],
        0.0,
        size[0],
        size[1],
        size[2],
        bounding_box[4]
    ], dtype=float)

    return bounding_box


def get_bounding_box_translation(bounding_box: np.ndarray) -> Tuple[float, float, float]:
    """Returns the coordinate values of the bounding box center.

    Arguments:
        bounding_box: Bounding box in the format
            [center_x, center_y, center_z, width, length, height, theta_x]

    Returns:
        x: Bounding box center x position in m.
        y: Bounding box center y position in m.
        z: Bounding box center z position in m.
    """
    x = bounding_box[0]
    y = bounding_box[1]
    z = bounding_box[2]
    return x, y, z


def get_bounding_box_rotation(bounding_box: np.ndarray) -> Tuple[float, float, float, float]:
    """Returns the values of the bounding box rotation as quaternion.

    Arguments:
        bounding_box: Bounding box in the format
            [center_x, center_y, center_z, width, length, height, theta_x]

    Returns:
        w: Bounding box orientation w-value of the quaternion.
        x: Bounding box orientation x-value of the quaternion.
        y: Bounding box orientation y-value of the quaternion.
        z: Bounding box orientation z-value of the quaternion.
    """
    # Transform bounding box rotation to a quaternion
    quaternion = Quaternion(axis=[0, 0, 1], angle=bounding_box[-1])

    return quaternion.w, quaternion.x, quaternion.y, quaternion.z


def get_bounding_box_size(bounding_box: np.ndarray, detection_name: str) -> Tuple[float, float, float]:
    """Returns the 3D size of a given 2D bounding box and class.

    Arguments:
        bounding_box: Bounding box in the format
            [center_x, center_y, length, width, theta_x]
        detection_name: Name of the class according to
            the nuScenes detection challenge.

    Returns:
        w: Bounding box width in m.
        l: Bounding box length in m.
        h: Bounding box height in m.
    """
    # Define class specific height map
    height_map: Dict[str, float] = {
        'void': 1.029,
        'barrier': 0.981,
        'bicycle': 1.283,
        'bus': 3.41,
        'car': 1.698,
        'construction_vehicle': 3.05,
        'motorcycle': 1.471,
        'pedestrian': 1.78,
        'traffic_cone': 1.067,
        'trailer': 4.04,
        'truck': 2.843
    }

    # Set bounding box heigt
    w = float(bounding_box[3])
    l = float(bounding_box[2])
    h = float(height_map[detection_name])

    return w, l, h


def get_bounding_box_velocity(velocity: np.ndarray, nusc: nuscenes.NuScenes, sample_token: str) -> Tuple[float, float]:
    """
    Arguments:
        velocity: Velocity vector of the associated radar point
            in the vehicle frame.
        nusc: nuScenes instance.
        sample_token: Token of the corresponding nuScenes sample.

    Returns:
        vx: Velocity in x in the global frame.
        vy: Velocity in y in the global frame.
    """
    # Split velocity in x and y-direction
    vx = 0.0
    vy = 0.0

    return vx, vy


def get_bounding_box_detection_name(label: int) -> str:
    """ Returns the nuScenes detection name according to a given label.

    Arguments:
        label: Numerical class label.

    Returns:
        detection_name: Name of the class according to
            the nuScenes detection challenge.
    """
    detection_name_list: List[str] = [
        'void',
        'barrier',
        'bicycle',
        'bus',
        'car',
        'construction_vehicle',
        'motorcycle',
        'pedestrian',
        'traffic_cone',
        'trailer',
        'truck'
    ]
    return detection_name_list[int(label)]


def get_bounding_box_attribute_name(detection_name: str, velocity: np.ndarray) -> str:
    """ Returns the nuScenes attribute name for a given detection name.

    Possible attribute names for the different classes of the nuScenes
    detection challenge:
    barrier:	            void
    traffic_cone:	        void
    bicycle:	            cycle.{with_rider, without_rider}
    motorcycle:	            cycle.{with_rider, without_rider}
    pedestrian:	            pedestrian.{moving, standing, sitting_lying_down}
    car:	                vehicle.{moving, parked, stopped}
    bus:	                vehicle.{moving, parked, stopped}
    construction_vehicle:	vehicle.{moving, parked, stopped}
    trailer:	            vehicle.{moving, parked, stopped}
    truck:              	vehicle.{moving, parked, stopped}

    Arguments:
        detection_name: Name of the class according to
            the nuScenes detection challenge.
        velocity: Velocity vector of the associated radar point.

    Returns:
        attribute_name: A corresponding attribute name to
            the given detection name.
    """
    attribute_name_map: Dict[str, str] = {
        'barrier': '',
        'traffic_cone': '',
        'bicycle': 'cycle.with_rider',
        'motorcycle': 'cycle.with_rider',
        'pedestrian': 'pedestrian.moving',
        'car': 'vehicle.moving',
        'bus': 'vehicle.moving',
        'construction_vehicle': 'vehicle.moving',
        'trailer': 'vehicle.moving',
        'truck': 'vehicle.moving'
    }

    return attribute_name_map[detection_name]


def get_sample_token(graph_name: str) -> str:
    """ Returns the nuScenes sample token from a given graph file name.

    Assumes that the graph file name has the following pattern:
    /.../..._sample_token.pt

    Arguments:
        graph_name: Name of the graph file.

    Returns:
        sample_token: nuScenes sample token.
    """
    file_name, _ = os.path.splitext(os.path.split(graph_name)[-1])
    sample_token = file_name.split('_')[-1]
    return sample_token


def convert_results(nusc: nuscenes.NuScenes, bb_preds: Dict[str, List], vels: List[np.ndarray], graph_names: List[str]) -> Dict:
    """ Converts the bounding box prediction to the nuScenes bounding box submission format.

    Arguments:
        nusc: nuScenes instance for the selected dataset split.
        bb_preds: Bounding box prediction.
        vels: Velocity values of the radar points (nodes).
        graph_names: Names of the individual input graphs.

    Returns:
        results: Dictionary of lists containing the predicted bounding boxes
            of the samples in the nuScenes bounding box submission format.
    """
    # Check compatibility
    assert len(bb_preds) == len(graph_names) == len(vels)

    # Create results dictionary
    results: Dict[str, List] = {}

    for bb_pred, vel, graph_name in zip(bb_preds, vels, graph_names):
        # Convert sample results
        sample_token = get_sample_token(graph_name)
        results[sample_token]: List[Dict] = []

        # Convert bounding box to absolute rotated bounding box
        if bb_pred['boxes']:
            bb_pred['boxes'] = bb_pred['boxes'][0].get_absolute_rotated_box_representations(bb_pred['boxes'])

        for i, bb in enumerate(bb_pred['boxes']):
            # Get bounding box properties in nuScenes format
            detection_name = get_bounding_box_detection_name(bb_pred['labels'][i])
            detection_score = float(bb_pred['scores'][i])
            attribute_name = get_bounding_box_attribute_name(detection_name, vel[i])
            size = get_bounding_box_size(bb, detection_name)

            # Convert 2D bounding box to 3D
            bb = get_3d_bounding_box(bb, size)

            # Transform bounding box from vehicle to gloabl frame
            bb = transform_bounding_box(bb, nusc, sample_token)

            # Set center point z value to height/2
            bb[2] += size[2] / 2

            # Get geometric bounding box properties in nuScenes format
            translation = get_bounding_box_translation(bb)
            rotation = get_bounding_box_rotation(bb)

            # Get bounding box velocity
            velocity = get_bounding_box_velocity(vel[i], nusc, sample_token)

            # Convert bounding box
            results[sample_token].append({
                "sample_token": sample_token,
                "translation": translation,
                "size": size,
                "rotation": rotation,
                "velocity": velocity,
                "detection_name": detection_name,
                "detection_score": detection_score,
                "attribute_name": attribute_name
            })

    return results


def get_submission(nusc: nuscenes.NuScenes, bb_pred: Dict[str, List], vel: List[np.ndarray], graph_names: List[str]) -> Dict:
    """ Converts the model prediction to the nuScenes submission format.

    Arguments:
        nusc: nuScenes instance for the selected dataset split.
        bb_pred: Bounding box prediction.
        vel: Velocity values of the radar points (nodes).
        graph_names: Names of the individual input graphs.

    Returns:
        submission: Dictionary of model predictions in the nuScenes object
            detection submission format.
    """
    # Create a copy of the bounding box predictions
    bb_preds = copy.deepcopy(bb_pred)
    vels = copy.deepcopy(vel)

    # Create submission dictionary
    submission = {
        "meta": {
            "use_camera": False,
            "use_lidar": False,
            "use_radar": True,
            "use_map": False,
            "use_external": False,
        }
    }

    # Create and insert results
    submission["results"] = convert_results(nusc, bb_preds, vels, graph_names)

    return submission
