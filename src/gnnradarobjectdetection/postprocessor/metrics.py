from typing import List

import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, multilabel_confusion_matrix

from gnnradarobjectdetection.postprocessor.configs import PostProcessingConfiguration
from gnnradarobjectdetection.postprocessor.torchmetrics_mean_ap import MeanAveragePrecision
from gnnradarobjectdetection.preprocessor.bounding_box import BoundingBox


class ObjectDetectionMetrics():
    """ Groups methods for calculating common object detection metrics.

    Methods:
        get_map: Calculates mAP based on model predictions and ground truth.
    """

    @staticmethod
    def __get_prediction_dict_list_for_mAP(bb_pred: List) -> List:
        """ Transforms bounding box predictions to a common representation.

        For aligned boxes, the final format of each box is: [x_min, y_min, x_max, y_max]
        For rotated boxes, the final format of each box is: [x_center, y_center, length, width, theta]

        Arguments:
            bb_pred: List of predicted bounding boxes for multiple graphs.

        Returns:
            prediction_dicts: List of dicts. Each dict contains all predicted boxes, their label and confidence scores of one graph.
        """

        prediction_dicts = []

        for object_detection_result in bb_pred:
            prediction_dict = {}

            if len(object_detection_result["boxes"]) == 0:
                bounding_box_matrix = np.empty([0, 0])
            elif object_detection_result["boxes"][0].is_aligned:
                bounding_box_matrix = BoundingBox.get_two_point_representations(object_detection_result.get("boxes"))
            elif object_detection_result["boxes"][0].is_rotated:
                bounding_box_matrix = BoundingBox.get_absolute_rotated_box_representations(object_detection_result.get("boxes"))

            prediction_dict["boxes"] = torch.tensor(bounding_box_matrix, dtype=torch.float32)
            prediction_dict["scores"] = torch.tensor(object_detection_result.get("scores"), dtype=torch.float32)
            prediction_dict["labels"] = torch.tensor(object_detection_result.get("labels"), dtype=torch.long)

            prediction_dicts.append(prediction_dict)

        return prediction_dicts

    @staticmethod
    def __get_ground_truth_dict_list_for_mAP(bb_ground_truth: List) -> List:
        """ Transforms ground truth bounding box to a common representation.

        For aligned boxes, the final format of each box is: [x_min, y_min, x_max, y_max]
        For rotated boxes, the final format of each box is: [x_center, y_center, length, width, theta]

        Arguments:
            bb_pred: List of ground truth bounding boxes for multiple graphs.

        Returns:
            prediction_dicts: List of dicts. Each dict contains all ground truth boxes and their label of one graph.
        """

        ground_truth_dicts = []

        for ground_truth_objects in bb_ground_truth:
            ground_truth_dict = {}

            if len(ground_truth_objects["boxes"]) == 0:
                bounding_box_matrix = np.empty([0, 0])
            elif ground_truth_objects["boxes"][0].is_aligned:
                bounding_box_matrix = BoundingBox.get_two_point_representations(ground_truth_objects.get("boxes"))
            elif ground_truth_objects["boxes"][0].is_rotated:
                bounding_box_matrix = BoundingBox.get_absolute_rotated_box_representations(ground_truth_objects.get("boxes"))

            ground_truth_dict["boxes"] = torch.tensor(bounding_box_matrix, dtype=torch.float32)
            ground_truth_dict["labels"] = torch.tensor(ground_truth_objects.get("labels"), dtype=torch.long)

            ground_truth_dicts.append(ground_truth_dict)

        return ground_truth_dicts

    @classmethod
    def get_map(cls, eval_config: PostProcessingConfiguration, bb_pred: List, bb_ground_truth: List, cls_pred: List) -> dict:
        """ Calculates the mAP of the model.

        Arguments:
            eval_config: Evaluation configuration.
            bb_pred: List with processed bounding box predictions for multiple graphs.
            bb_ground_truth: List with ground truth bounding boxes for multiple graphs.
            cls_pred: Predicted class labels for multiple graphs.

        Returns:
            res: dict containing
                - map: ``torch.Tensor``
                - map_per_class: ``torch.Tensor``
        """

        iou_thresholds = [eval_config.iou_for_mAP]

        # get positions
        pos = [d['pos'] for d in cls_pred]

        # check if bounding boxes are rotated or aligned
        aligned = bb_pred[0]['boxes'][0].is_aligned

        print(">>> postprocess raw box predictions")
        prediction_dicts = cls.__get_prediction_dict_list_for_mAP(bb_pred)
        ground_truth_dicts = cls.__get_ground_truth_dict_list_for_mAP(bb_ground_truth)

        # so far only Point-IOU based mAP calculation is possible for rotated bounding boxes
        if not aligned and not eval_config.use_point_iou:
            raise Exception("so far only Point-IOU based mAP calculation is possible for rotated bounding boxes, \
                             select 'use_point_iou = True' in configuration for rotated boxes")
        elif not aligned:
            print(">>> calculate mAP for rotated bounding boxes")
        elif aligned:
            print(">>> calculate mAP for aligned bounding boxes")

        mean_average_precision = MeanAveragePrecision(
            "xyxy", "bbox", iou_thresholds, class_metrics=True)

        if eval_config.use_point_iou:
            mean_average_precision.update(prediction_dicts, ground_truth_dicts, True, pos, aligned)
        else:
            mean_average_precision.update(prediction_dicts, ground_truth_dicts)

        res = mean_average_precision.compute()

        return res


class SegmentationMetrics():
    """ Groups methods for calculating common semantic segmentation metrics.

    Methods:
        get_f1: Calculates f1 score.
        get_confusion_matrix: Calculate confusion matrix.
        get_confusion_matrices_per_class: Calculate class individual confusion matrices.
    """

    def __init__(self, cls_pred_label: List, cls_ground_truth: List):
        """
        Arguments:
            cls_pred_label: List with class label predictions of all nodes for multiple graphs.
            cls_ground_truth: List with ground truth class labels of all nodes for multiple graphs.
        """

        self.y_true = self.__get_ground_truth_vector(cls_ground_truth)
        self.y_pred = self.__get_prediction_vector(cls_pred_label)

    def get_f1(self, num_classes: int, average: str) -> list:
        f1_list = f1_score(self.y_true, self.y_pred, labels=range(num_classes), average=average)
        return f1_list

    def get_confusion_matrix(self, num_classes: int) -> np.ndarray:
        matrix = confusion_matrix(self.y_true, self.y_pred, labels=range(num_classes))
        return matrix

    def get_confusion_matrices_per_class(self, num_classes: int) -> np.ndarray:
        matrix = multilabel_confusion_matrix(self.y_true, self.y_pred, labels=range(num_classes))
        return matrix

    @staticmethod
    def __get_prediction_vector(cls_pred_label) -> list:

        y_pred = None

        for cls_pred_class in cls_pred_label:
            if y_pred is None:
                y_pred = cls_pred_class
            else:
                y_pred = np.concatenate((y_pred, cls_pred_class), axis=0)

        y_pred = y_pred.astype(int)
        return y_pred.tolist()

    @staticmethod
    def __get_ground_truth_vector(cls_ground_truth) -> list:

        y_true = None

        for cls_true in cls_ground_truth:
            cls_true = cls_true['labels'].reshape(cls_true['labels'].shape[0], 1)

            if y_true is None:
                y_true = cls_true
            else:
                y_true = np.concatenate((y_true, cls_true), axis=0)

        y_true = y_true.astype(int)

        return y_true.tolist()
