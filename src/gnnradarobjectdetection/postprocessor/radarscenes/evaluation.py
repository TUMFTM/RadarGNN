import os
import json
from dataclasses import asdict
import numpy as np

from gnnradarobjectdetection.postprocessor.evaluation import Evaluator, get_new_evaluation_folder_path
from gnnradarobjectdetection.postprocessor.configs import PostProcessingConfiguration
from gnnradarobjectdetection.postprocessor.metrics import SegmentationMetrics, ObjectDetectionMetrics
from gnnradarobjectdetection.postprocessor.visualization import plot_confusion_matrix


class RadarscenesEvaluator(Evaluator):

    def __init__(self, config: PostProcessingConfiguration, *args, **kwargs):
        # Initialize radarscenes evaluator
        self.mAP = None
        self.mAP_per_class = None
        self.f1_segmentation = None
        self.confusion_absolute = None
        self.confusion_relative = None

        # Initialize base class
        super().__init__(config, *args, **kwargs)

    def evaluate(self, bb_pred, bb_ground_truth, cls_pred, cls_pred_label, cls_ground_truth, *args, **kwargs) -> None:
        """ Evaluates the model performance based on the model prediction and ground truth information.

        Arguments:
            bb_pred: Predicted bounding boxes.
            bb_ground_truth: Ground truth bounding boxes.
            cls_pred: Numerical class value prediction.
            cls_pred_label: Class names for the class values.
            cls_ground_truth: Ground truth class value.
        """
        if self.config.get_mAP:
            mAP_res = ObjectDetectionMetrics.get_map(self.config, bb_pred, bb_ground_truth, cls_pred)
            self.mAP = mAP_res.get('map').item()
            self.mAP_per_class = mAP_res.get('map_per_class').detach().numpy()

        segmentation_metric = SegmentationMetrics(cls_pred_label, cls_ground_truth)
        if self.config.get_segmentation_f1:
            self.f1_segmentation = segmentation_metric.get_f1(len(self.names), self.config.f1_class_averaging)

        if self.config.get_confusion:
            self.confusion_absolute = segmentation_metric.get_confusion_matrix(len(self.names))

            # process confusion matrix
            sums = self.confusion_absolute.astype(np.float).sum(axis=1).reshape(self.confusion_absolute.shape[0], 1)
            sums[np.where(sums == 0)[0]] = 1E-8
            self.confusion_relative = self.confusion_absolute / sums

    def save_results(self, path_to_model_folder, *args, **kwargs):
        """ Saves the evaluation results."""
        next_folder_path = get_new_evaluation_folder_path(path_to_model_folder)
        os.mkdir(next_folder_path)

        # create dictionaries that can then be written into Json file
        eval_config_dict = asdict(self.config)
        json_dict = {"EVALUATION_CONFIG": eval_config_dict}

        # write JSON file
        json_path = f"{next_folder_path}/eval_configs.json"
        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=4)

        # create results dict and save it
        detection_results = {}
        if self.config.get_mAP:
            detection_results["mAP"] = self.mAP
            detection_results["mAP_per_class"] = self.mAP_per_class.tolist()

        segmentation_results = {}
        if self.config.get_segmentation_f1:
            if type(self.f1_segmentation) == np.ndarray:
                segmentation_results["f1"] = self.f1_segmentation.tolist()
            else:
                segmentation_results["f1"] = self.f1_segmentation

        json_dict = {"OBJECT_DETECTION_METRICS": detection_results,
                     "SEMANTIC_SEGMENTATION_METRICS": segmentation_results}
        json_path = f"{next_folder_path}/eval_results.json"

        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=4)

        if self.config.get_confusion:
            # save confusion matrices
            with open(f"{next_folder_path}/confusion_abs.npy", 'wb') as f:
                np.save(f, self.confusion_absolute)

            with open(f"{next_folder_path}/convusion_rel.npy", 'wb') as f:
                np.save(f, self.confusion_relative)

            # create confusion matrix plots and save them
            conf_perc = np.round((self.confusion_relative * 100), 2)
            fig = plot_confusion_matrix(conf_perc, self.names, normalize=False)
            fig.savefig(f"{next_folder_path}/confusion.png")
