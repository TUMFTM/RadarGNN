
import numpy as np
import matplotlib.pyplot as plt
import itertools

from gnnradarobjectdetection.preprocessor.radar_point_cloud import RadarPointCloud
from gnnradarobjectdetection.utils.radar_scenes_properties import Labels


class PredictionVisualizer():

    @staticmethod
    def show_final_model_prediction(object_detection_result: dict, semantic_segmentation_result: dict,
                                    box_with_labels: bool = True):
        """ Visualizes the model prediction.

        Arguments:
            object_detection_result: Predicted bounding boxes.
            semantic_segmentation_result: Predicted class labels.
            box_with_labels: Wether to show the bonding box class.
        """
        pos = semantic_segmentation_result.get("pos")
        labels = semantic_segmentation_result.get("labels")

        bounding_boxes = object_detection_result.get("boxes")
        box_scores = object_detection_result.get("scores")
        box_labels = object_detection_result.get("labels")

        if box_with_labels:
            _, ax = Visualizer.show_point_cloud_with_bounding_boxes(
                pos, labels, bounding_boxes, box_labels, box_scores)
        else:
            _, ax = Visualizer.show_point_cloud_with_bounding_boxes(
                pos, labels, bounding_boxes)
        ax.set_title("Final GNN prediction")

    @staticmethod
    def show_ground_truth(ground_truth_objects: dict, ground_truth_segmentation: dict,
                          box_with_labels: bool = True, show_velocity_vector=True):
        """ Visualizes the ground truth data.

        Arguments:
            ground_truth_objects: Ground truth bounding boxes.
            ground_truth_segmentation: Ground truth class labels.
            box_with_labels: Wether to show the bonding box class.
            show_velocity_vector: Wether to show the velocity vector.
        """

        pos = ground_truth_segmentation.get("pos")
        vel = ground_truth_segmentation.get("vel")
        labels = ground_truth_segmentation.get("labels")

        bounding_boxes = ground_truth_objects.get("boxes")
        box_labels = ground_truth_objects.get("labels")

        if box_with_labels:
            _, ax = Visualizer.show_point_cloud_with_bounding_boxes(
                pos, labels, bounding_boxes, box_labels, None, show_velocity_vector, vel)
        else:
            _, ax = Visualizer.show_point_cloud_with_bounding_boxes(
                pos, labels, bounding_boxes, None, None, show_velocity_vector, vel)

        ax.set_title("Ground truth")


class Visualizer():

    @staticmethod
    def show_point_cloud_with_bounding_boxes(x, labels, bounding_boxes, box_labels=None,
                                             box_scores=None, show_velocity_vector=False, vel=None):
        """ Visualizes a given point cloud and bounding boxes.

        Arguments:
            x: Point cloud.
            labels: Class labels.
            bounding_boxes: Bounding boxes.
            box_scores: Bounding box confidence scores.
            show_velocity_vector: Wether to show the velocity vector.
        """

        point_cloud = RadarPointCloud()
        point_cloud.X_cc = x
        point_cloud.V_cc_compensated = vel
        point_cloud.label_id = labels.reshape(labels.shape[0], 1)

        fig, ax = point_cloud.show(show_velocity_vector=show_velocity_vector)

        label_dict = Labels.get_label_dict()

        for i, bb in enumerate(bounding_boxes):
            bb.plot_on_axis(ax)

            # if label and score is available -> Plot into the box
            if box_scores is not None or box_labels is not None:

                if box_scores is not None and box_labels is not None:
                    score = box_scores[i]
                    label = box_labels[i]
                    txt = f"{round(score * 100)}% - {label_dict.get(label)}"

                elif box_scores is None and box_labels is not None:
                    label = box_labels[i]
                    txt = f"{label_dict.get(label)}"

                x = np.min(bb.corners[:, 0])
                y = np.max(bb.corners[:, 1]) + 0.1

                ax.text(x, y, txt, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'), fontsize="xx-small")

        return fig, ax


def plot_confusion_matrix(cm: np.ndarray,
                          target_names: list,
                          title: str = 'Confusion matrix',
                          cmap=None,
                          normalize: bool = True):
    """ Visualizes a given confusion matrix.

        Arguments:
            cm: Confusion matrix.
            target_names: Class names.
            title: Plot title.
            cmap: Color map.
            normalize: Wether to normalize the data.
        """

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig
