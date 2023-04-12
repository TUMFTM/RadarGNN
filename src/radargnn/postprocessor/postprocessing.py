from typing import Dict, List, Tuple

import torchvision
import torch
import numpy as np
from detectron2.layers import nms_rotated
from sklearn.neighbors import kneighbors_graph

from radargnn.postprocessor.configs import PostProcessingConfiguration
from radargnn.preprocessor.bounding_box import BoundingBox, RelativeAlignedBoundingBox, RelativeRotatedBoundingBox, RotationInvariantRelativeRotatedBoundingBox, AbsoluteRotatedBoundingBox, invert_bb_orientation_angle_adaption


class Postprocessor():
    """ Groups methods for postprocessing the raw GNN predictions.

    Methods:
        process_one_raw_prediction: Postprocesses raw GNN prediction for one graph.
        process_one_ground_truth: Postprocesses the ground truth for one graph.
        process: Postprocesses ground truth and predictions for multiple graphs.
    """

    @staticmethod
    def process_one_raw_prediction(config: PostProcessingConfiguration, pos: np.ndarray,
                                   raw_bb_pred: np.ndarray, raw_cls_prob_pred: np.ndarray) -> Tuple[Dict]:
        """ Postprocesses raw GNN prediction for one graph.

        Transforms all bounding boxes to their absolute representation defined by the absolute coordinates of the corners.
        Applies a non-maximum-suppression to remove overlapping box predictions.
        Stores the final object detections in a dict.
        Reduces the predicted probability distribution to a final class label prediction for each node.
        Additionally, extracts the probability of the final class prediction and the background probability.
        Stores the final segmentation results in a dict.

        Arguments:
            config: Postprocessing configuration.
            pos: Spatial coordinates of all nodes of one graph.
            raw_bb_pred: Raw bounding box predictions of all nodes of one graph.
            raw_cls_prob_pred: Raw class label predictions of all nodes of one graph.

        Returns:
            object_detection_result: Dict with processed bounding box predictions for one graph.
            semantic_segmentation_result: Dict with processed class predictions for one graph.
        """

        object_detection_result = {}
        semantic_segmentation_result = {}

        # process raw object proposals

        # suppresses all background bounding boxes and boxes with bad score + transforms raw outputs to BoundingBox object
        bounding_boxes, box_scores, box_labels = PredictionExtractor.get_absolute_object_bounding_box_predictions(
            raw_cls_prob_pred, raw_bb_pred, pos, config)

        # applies non-maximum-suppression
        bounding_boxes, box_scores, box_labels = BoxSuppressor.apply_nms(
            bounding_boxes, box_scores, box_labels, config.iou_for_nms)

        # store results in a dict
        object_detection_result["boxes"] = bounding_boxes
        object_detection_result["scores"] = box_scores[:, 0]
        object_detection_result["labels"] = box_labels[:, 0]

        # process class probability predictions
        semantic_segmentation_result["pos"] = pos
        semantic_segmentation_result["labels"] = PredictionExtractor.get_predicted_label(
            raw_cls_prob_pred)[:, 0]
        semantic_segmentation_result["scores"] = PredictionExtractor.get_prediction_scores(
            raw_cls_prob_pred)[:, 0]
        semantic_segmentation_result["clutter_scores"] = PredictionExtractor.get_clutter_scores(
            raw_cls_prob_pred, config.bg_index)[:, 0]

        return object_detection_result, semantic_segmentation_result

    @staticmethod
    def process_one_ground_truth(pos: np.ndarray, vel: np.ndarray, raw_bb_ground_truth: np.ndarray,
                                 raw_cls_ground_truth: np.ndarray, bb_invariance: str, bg_index: int) -> tuple:
        """ Postprocesses ground truth for one graph.

        Transforms all bounding boxes to their absolute representation defined by the absolute coordinates of the corners.
        Removes all identical boxes -> Keeps only one bounding box per object.
        Stores the final true object boxes in a dict.
        Stores the ground truth segmentation in a dict.

        Arguments:
            pos: Spatial coordinates of all nodes of one graph.
            vel: Velocity vector of all nodes of one graph.
            raw_bb_ground_truth: Raw true bounding boxes of all nodes of one graph.
            raw_cls_ground_truth: Raw true class labels of all nodes of one graph.
            bb_invariance: Bounding box representation.
            bg_index: Class index of the background class.

        Returns:
            ground_truth_objects: Dict with processed bounding boxes for one graph.
            ground_truth_segmentation: Dict with processed class labels for one graph.
        """

        ground_truth_objects = {}
        ground_truth_segmentation = {}

        # process ground truth bounding boxes
        # transform relative ground truth bounding box matrix to a list of absolute BoundingBox objects with its class labels
        bounding_boxes, box_labels = GroundTruthExtractor.get_absolute_object_bounding_boxes(
            raw_cls_ground_truth, raw_bb_ground_truth, pos, bb_invariance, bg_index)

        # removes all duplicate bounding boxes
        bounding_boxes, box_labels = GroundTruthExtractor.remove_duplicate_boxes(
            bounding_boxes, box_labels)

        ground_truth_objects["boxes"] = bounding_boxes
        ground_truth_objects["labels"] = box_labels[:, 0]

        # process ground truth point labels
        ground_truth_segmentation["pos"] = pos
        ground_truth_segmentation["vel"] = vel
        ground_truth_segmentation["labels"] = raw_cls_ground_truth

        return ground_truth_objects, ground_truth_segmentation

    def process(self, config: PostProcessingConfiguration,
                raw_pos: np.ndarray, raw_vel: np.ndarray,
                predictions: Dict, ground_truth: Dict) -> tuple:
        """ Postprocesses ground truth and predictions for multiple graphs.

        Arguments:
            config: Postprocessing configuration.
            raw_pos: Spatial coordinates of all nodes of multiple graph.
            raw_vel: Velocity vector of all nodes of multiple graph.
            predictions: Raw predictions of multiple graph.
            ground_truth: Ground truth of multiple graph.

        Returns:
            bb_pred: List of dicts with processed bounding box predictions.
            bb_ground_truth: List of dicts with processed ground truth bounding boxes.
            cls_pred: List of dicts with processed class label predictions.
            cls_ground_truth: List of dicts with processed ground truth class labels.
        """

        # initialize return values
        bb_pred = []
        bb_ground_truth = []
        cls_pred = []
        cls_ground_truth = []

        # extract raw values
        raw_bb_pred = predictions.get("bounding_box_predictions")
        raw_cls_prob_pred = predictions.get("class_probability_prediction")
        raw_bb_ground_truth = ground_truth.get("bounding_box_true")
        raw_cls_ground_truth = ground_truth.get("class_true")

        # post process raw predictions
        for pos_raw, bb_raw, cls_raw in zip(raw_pos, raw_bb_pred, raw_cls_prob_pred):
            bb, cls = self.process_one_raw_prediction(config, pos_raw, bb_raw, cls_raw)
            bb_pred.append(bb)
            cls_pred.append(cls)

        # post process raw ground truth
        for pos_raw, vel_raw, bb_gt_raw, cls_gt_raw in zip(raw_pos, raw_vel, raw_bb_ground_truth, raw_cls_ground_truth):
            bb_gt, cls_gt = self.process_one_ground_truth(pos_raw, vel_raw, bb_gt_raw, cls_gt_raw, config.bb_invariance, config.bg_index)
            bb_ground_truth.append(bb_gt)
            cls_ground_truth.append(cls_gt)

        return bb_pred, bb_ground_truth, cls_pred, cls_ground_truth


class PredictionExtractor():
    """ Groups methods for postprocessing model predictions.

    Methods:
        get_predicted_label: Returns final class label predictions.
        get_prediction_scores: Returns probability of final class prediction.
        get_clutter_scores: Returns probability of being background.
        get_absolute_object_bounding_box_predictions: Transforms predicted boxes to an standardized representation.
    """

    @staticmethod
    def get_predicted_label(class_probability_prediction: np.ndarray) -> np.ndarray:

        labels = np.zeros([class_probability_prediction.shape[0], 1])
        for i in range(class_probability_prediction.shape[0]):
            vec = class_probability_prediction[i, :]
            labels[i, 0] = int(np.where(vec == np.max(vec))[0][0])

        return labels

    @staticmethod
    def get_prediction_scores(class_probability_prediction: np.ndarray) -> np.ndarray:
        scores = np.zeros([class_probability_prediction.shape[0], 1])
        for i in range(class_probability_prediction.shape[0]):
            vec = class_probability_prediction[i, :]
            scores[i, 0] = np.max(vec)

        return scores

    @staticmethod
    def get_clutter_scores(class_probability_prediction: np.ndarray, bg_index: int) -> np.ndarray:
        return class_probability_prediction[:, bg_index].reshape(class_probability_prediction.shape[0], 1)

    @classmethod
    def get_absolute_object_bounding_box_predictions(cls, class_probability_prediction: np.ndarray, bounding_box_predictions: np.ndarray,
                                                     pos: np.ndarray, config: PostProcessingConfiguration):

        """ Transforms raw object predictions of one graph to a standardized output.

        Arguments:
            class_probability_prediction: Matrix with raw class probability predictions of all nodes of one graph.
            bounding_box_predictions: Matrix with raw bounding box predictions of all nodes of one graph.
            pos: Spatial coordinates of all nodes of one graph.
            config: Postprocessing configuration.

        Returns:
            bounding_boxes: List with absolute bounding boxes (defined by their corners) of each node.
            box_scores: List of corresponding confidence scores (class probability) of each node.
            box_labels: List of corresponding class labels of each node.
        """

        # get labels and scores based on predictions
        labels = cls.get_predicted_label(class_probability_prediction)
        scores = cls.get_prediction_scores(class_probability_prediction)
        clutter_score = cls.get_clutter_scores(class_probability_prediction, config.bg_index)

        # filter out all bounding boxes from wrong predictions (with background class probability > clutter score)
        idx_remove_1 = np.where(clutter_score >= config.max_score_for_background)[0]
        idx_remove_2 = np.where(labels == config.bg_index)[0]
        idx_remove = np.concatenate((idx_remove_1, idx_remove_2), axis=0)

        # get for each class indices to remove based on the classes min object score
        for i, min_score in enumerate(config.min_object_score.values()):
            idx_remove = np.concatenate((idx_remove, np.where((scores <= min_score) & (labels == i))[0]), axis=0)

        idx_remove = np.unique(idx_remove)

        # get nearest neighbor of each point and then remove nn points of non boxes
        if config.bb_invariance == "en" and pos.shape[0] != 0:
            A_sparse = kneighbors_graph(pos, 1, mode='connectivity', include_self=False)
            A_full = A_sparse.toarray()
            pos_nn = pos[np.where(A_full == 1)[1]]
            pos_nn = np.delete(pos_nn, idx_remove, axis=0)

        # remove entries of non boxes
        bounding_box_predictions = np.delete(bounding_box_predictions, idx_remove, axis=0)
        pos = np.delete(pos, idx_remove, axis=0)
        box_scores = np.delete(scores, idx_remove, axis=0)
        box_labels = np.delete(labels, idx_remove, axis=0)

        # transform relative bounding boxes to absolute bounding boxes
        bounding_boxes = []

        for i in range(bounding_box_predictions.shape[0]):

            if bounding_box_predictions.shape[1] == 4:
                # aligned bounding box

                # extract bounding box information
                bb = bounding_box_predictions[i, :].reshape(1, 4)
                x_rel = bb[0, 0]
                y_rel = bb[0, 1]
                dx = bb[0, 2]
                dy = bb[0, 3]

                # get corresponding point coordinates
                point_koord = pos[i, :].reshape(1, 2)

                # transform relative bounding box first to absolute bounding box
                bb_relative = RelativeAlignedBoundingBox(x_rel, y_rel, dx, dy)
                bb_abs = bb_relative.get_absolute_bounding_box(point_koord[0, 0], point_koord[0, 1])
                bounding_boxes.append(bb_abs)

            else:
                # rotated bounding box

                if config.bb_invariance != "en":
                    bb = bounding_box_predictions[i, :].reshape(1, 5)
                    x_rel = bb[0, 0]
                    y_rel = bb[0, 1]
                    l = bb[0, 2]
                    w = bb[0, 3]

                    # if angle was shifted and passed through sin, invert these steps again
                    if config.adapt_orientation_angle:
                        theta = invert_bb_orientation_angle_adaption(bb[0, 4]) * 180 / np.pi
                    else:
                        theta = bb[0, 4] * 180 / np.pi  # transform from rad to degree as expected by RelativeRotatedBoundingBox

                    if config.bb_invariance == "translation":
                        # get corresponding point coordinates
                        point_koord = pos[i, :].reshape(1, 2)

                        # transform relative bounding box first to absolute bounding box
                        bb_relative = RelativeRotatedBoundingBox(x_rel, y_rel, l, w, theta)
                        bb_abs = bb_relative.get_absolute_bounding_box(point_koord[0, 0], point_koord[0, 1])

                    elif config.bb_invariance == "none":
                        # transform relative bounding box first to absolute bounding box
                        bb_relative = AbsoluteRotatedBoundingBox(x_rel, y_rel, l, w, theta)
                        bb_abs = bb_relative.get_absolute_bounding_box()

                    bounding_boxes.append(bb_abs)

                elif config.bb_invariance == "en":
                    # Rotation invariant representation of a rotated bounding box
                    bb = bounding_box_predictions[i, :].reshape(1, 5)
                    d = bb[0, 0]
                    theta_v_p_nn_v_p_c = bb[0, 1] * 180 / np.pi  # transform from rad to degree as expected by RotationInvariantRelativeRotatedBoundingBox
                    l = bb[0, 2]
                    w = bb[0, 3]
                    theta_v_p_nn_v_dir = bb[0, 4] * 180 / np.pi  # transform from rad to degree as expected by RotationInvariantRelativeRotatedBoundingBox

                    # get corresponding point coordinates
                    point_koord = pos[i, :]

                    # coordinates of the points nearest neighbor
                    nn_koord = pos_nn[i, :]

                    # transform bounding box representation to absolute bounding box
                    bb_relative = RotationInvariantRelativeRotatedBoundingBox(d, theta_v_p_nn_v_p_c, l, w, theta_v_p_nn_v_dir)
                    bb_abs = bb_relative.get_absolute_bounding_box(point_koord, nn_koord)
                    bounding_boxes.append(bb_abs)

        return bounding_boxes, box_scores, box_labels

    def extract(self, predictions: Dict) -> List:
        # Initialize return value
        cls_pred_label = []

        # Extract raw values
        raw_cls_prob_pred = predictions.get("class_probability_prediction")

        # Get predicted labels
        for cls_raw in raw_cls_prob_pred:
            cls = self.get_predicted_label(cls_raw)
            cls_pred_label.append(cls)

        return cls_pred_label


class BoxSuppressor():
    """ Groups methods for applying a non-maximum suppression with aligned and rotated bounding boxes.
    """

    @classmethod
    def apply_nms(cls, bounding_boxes: list, box_scores: np.ndarray, box_labels: np.ndarray, iou_nms: float):

        # only do something if at least one bounding box is present
        if len(bounding_boxes) > 0:

            if bounding_boxes[0].is_rotated:
                # rotated nms
                bounding_boxes, box_scores, box_labels = cls.__apply_nms_rotated(bounding_boxes, box_scores, box_labels, iou_nms)
            else:
                # aligned nms
                bounding_boxes, box_scores, box_labels = cls.__apply_nms_aligned(bounding_boxes, box_scores, box_labels, iou_nms)

        return bounding_boxes, box_scores, box_labels

    @staticmethod
    def __apply_nms_rotated(bounding_boxes: list, box_scores: np.ndarray, box_labels: np.ndarray, iou_nms: float):

        # get a bounding box matrix with the required format
        bounding_box_matrix = BoundingBox.get_absolute_rotated_box_representations(bounding_boxes)

        # shift all boxes if some negative values are present for the bb center
        shift = 0
        if np.min(bounding_box_matrix[:, :2]) < 0:
            shift = abs(np.min(bounding_box_matrix[:, :2])) + 100
            bounding_box_matrix[:, :2] = bounding_box_matrix[:, :2] + shift

        # apply nms
        bounding_box_matrix = torch.tensor(bounding_box_matrix, dtype=torch.float64)
        box_scores = torch.tensor(box_scores[:, 0], dtype=torch.float64)
        idx_keep = nms_rotated(bounding_box_matrix, box_scores, iou_nms)

        bounding_box_matrix = bounding_box_matrix[idx_keep].detach().numpy()
        box_scores = box_scores[idx_keep].detach().numpy()
        box_scores = box_scores.reshape(box_scores.shape[0], 1)
        box_labels = box_labels[idx_keep]
        box_labels = box_labels.reshape(box_labels.shape[0], 1)

        # shift remaining bounding boxes back
        if shift != 0:
            bounding_box_matrix[:, :2] = bounding_box_matrix[:, :2] - shift

        # keep all indices of the bounding box list
        bounding_box_keep = []
        for idx in idx_keep:
            bounding_box_keep.append(bounding_boxes[idx])

        return bounding_box_keep, box_scores, box_labels

    @staticmethod
    def __apply_nms_aligned(bounding_boxes: list, box_scores: np.ndarray, box_labels: np.ndarray, iou_nms: float):

        # get a bounding box matrix with the required format
        bounding_box_matrix = BoundingBox.get_two_point_representations(bounding_boxes)

        # shift all boxes if some negative values are present
        shift = 0
        if np.min(bounding_box_matrix) < 0:
            shift = abs(np.min(bounding_box_matrix)) + 100
            bounding_box_matrix = bounding_box_matrix + shift

        # apply nms
        bounding_box_matrix = torch.tensor(bounding_box_matrix, dtype=torch.float32)
        box_scores = torch.tensor(box_scores[:, 0], dtype=torch.float32)

        idx_keep = torchvision.ops.nms(bounding_box_matrix, box_scores, iou_nms)

        bounding_box_matrix = bounding_box_matrix[idx_keep].detach().numpy()
        box_scores = box_scores[idx_keep].detach().numpy()
        box_scores = box_scores.reshape(box_scores.shape[0], 1)
        box_labels = box_labels[idx_keep]
        box_labels = box_labels.reshape(box_labels.shape[0], 1)

        # shift remaining bounding boxes back
        if shift != 0:
            bounding_box_matrix = bounding_box_matrix - shift

        # transform matrix back in to list of absolute bounding boxes
        bounding_boxes = []
        for i in range(bounding_box_matrix.shape[0]):
            bb = bounding_box_matrix[i, :]
            x_min = bb[0]
            x_max = bb[2]
            y_min = bb[1]
            y_max = bb[3]

            c1 = np.array([x_min, y_min]).reshape(1, 2)
            c2 = np.array([x_min, y_max]).reshape(1, 2)
            c3 = np.array([x_max, y_min]).reshape(1, 2)
            c4 = np.array([x_max, y_max]).reshape(1, 2)

            corners = np.concatenate((c1, c2, c3, c4), axis=0)

            bounding_boxes.append(BoundingBox(corners, True))

        return bounding_boxes, box_scores, box_labels


class GroundTruthExtractor():
    """ Groups methods for postprocessing ground truth data.

    Methods:
        get_absolute_object_bounding_boxes: Transforms predicted boxes to an standardized representation.
        remove_duplicate_boxes: Removes duplicate ground truth boxes and keeps only one bounding box per object.
    """

    @staticmethod
    def get_absolute_object_bounding_boxes(class_labels: np.ndarray, bounding_boxes: np.ndarray, pos: np.ndarray,
                                           bb_invariance: str, bg_index: int):
        """ Transforms ground truth bounding boxes of all nodes of one graph to a standardized output.

        Arguments:
            class_labels: Matrix with class labels of all nodes of one graph.
            bounding_boxes: Matrix with raw ground truth bounding box of all nodes of one graph.
            pos: Spatial coordinates of all nodes of one graph.
            bb_invariance: Bounding box representation.
            bg_index: Class index of the background class.

        Returns:
            bounding_boxes: List with absolute bounding boxes (defined by their corners) of each node.
            box_scores: List of corresponding confidence scores (class probability) of each node.
            box_labels: List of corresponding class labels of each node.
        """
        # filter out all nan-boxes from background
        idx_remove = np.where(class_labels == bg_index)[0]
        idx_remove = idx_remove

        # get nearest neighbor of each point and then remove nn points of non boxes
        if bb_invariance == "en" and pos.shape[0] != 0:
            A_sparse = kneighbors_graph(pos, 1, mode='connectivity', include_self=False)
            A_full = A_sparse.toarray()
            pos_nn = pos[np.where(A_full == 1)[1]]
            pos_nn = np.delete(pos_nn, idx_remove, axis=0)

        # remove all entries from non boxes
        bounding_boxes = np.delete(bounding_boxes, idx_remove, axis=0)
        pos = np.delete(pos, idx_remove, axis=0)
        box_labels = np.delete(class_labels, idx_remove, axis=0)
        box_labels = box_labels.reshape(box_labels.shape[0], 1)

        # transform relative bounding boxes to absolute bounding boxes
        bounding_box_list = []

        for i in range(bounding_boxes.shape[0]):

            if bounding_boxes.shape[1] == 4:
                # aligned bounding box

                # extract bounding box information
                bb = bounding_boxes[i, :].reshape(1, 4)
                x_rel = bb[0, 0]
                y_rel = bb[0, 1]
                dx = bb[0, 2]
                dy = bb[0, 3]

                # get corresponding point coordinates
                point_koord = pos[i, :].reshape(1, 2)

                # transform relative bounding box first to absolute bounding box and then to two point representation and store it
                bb_relative = RelativeAlignedBoundingBox(x_rel, y_rel, dx, dy)
                bb_abs = bb_relative.get_absolute_bounding_box(point_koord[0, 0], point_koord[0, 1])
                bounding_box_list.append(bb_abs)

            else:
                # rotated bounding box

                # Detect rotated bounding box via config for postprocessing
                if bb_invariance != "en":
                    # "Normal" rotated bounding box [x_rel, y_rel, l, w, theta_x (rad, 0-pi)]
                    bb = bounding_boxes[i, :].reshape(1, 5)
                    x_rel = bb[0, 0]
                    y_rel = bb[0, 1]
                    l = bb[0, 2]
                    w = bb[0, 3]
                    theta = bb[0, 4] * 180 / np.pi  # transform from rad to degree as expected by RelativeRotatedBoundingBox

                    if bb_invariance == "translation":
                        # get corresponding point coordinates
                        point_koord = pos[i, :].reshape(1, 2)

                        # transform relative bounding box first to absolute bounding box and then to two point representation and store it
                        bb_relative = RelativeRotatedBoundingBox(x_rel, y_rel, l, w, theta)
                        bb_abs = bb_relative.get_absolute_bounding_box(point_koord[0, 0], point_koord[0, 1])

                    elif bb_invariance == "none":
                        bb_relative = AbsoluteRotatedBoundingBox(x_rel, y_rel, l, w, theta)
                        bb_abs = bb_relative.get_absolute_bounding_box()

                    bounding_box_list.append(bb_abs)

                elif bb_invariance == "en":
                    # Rotation invariant representation of a rotated bounding box
                    bb = bounding_boxes[i, :].reshape(1, 5)
                    d = bb[0, 0]
                    theta_v_p_nn_v_p_c = bb[0, 1] * 180 / np.pi  # transform from rad to degree as expected by RotationInvariantRelativeRotatedBoundingBox
                    l = bb[0, 2]
                    w = bb[0, 3]
                    theta_v_p_nn_v_dir = bb[0, 4] * 180 / np.pi  # transform from rad to degree as expected by RotationInvariantRelativeRotatedBoundingBox

                    # get corresponding point coordinates
                    point_koord = pos[i, :]

                    # coordinates of the points nearest neighbor
                    nn_koord = pos_nn[i, :]

                    # transform bounding box representation to absolute bounding box
                    bb_relative = RotationInvariantRelativeRotatedBoundingBox(d, theta_v_p_nn_v_p_c, l, w, theta_v_p_nn_v_dir)
                    bb_abs = bb_relative.get_absolute_bounding_box(point_koord, nn_koord)
                    bounding_box_list.append(bb_abs)

        return bounding_box_list, box_labels

    @staticmethod
    def remove_duplicate_boxes(bounding_boxes: list, box_labels: np.ndarray):
        idx_remove = []

        # get all indices to remove
        for i in range(len(bounding_boxes)):
            corners = bounding_boxes[i].corners

            for j in range(i + 1, len(bounding_boxes)):
                corners_compare = bounding_boxes[j].corners

                if (corners == corners_compare).all() or (np.sum(abs(corners - corners_compare)) < 0.1):
                    idx_remove.append(j)

        idx_remove = list(set(idx_remove))

        # delete boxes and labels
        box_labels = np.delete(box_labels, idx_remove)
        box_labels = box_labels.reshape(box_labels.shape[0], 1)

        for idx in sorted(idx_remove, reverse=True):
            del bounding_boxes[idx]

        return bounding_boxes, box_labels
