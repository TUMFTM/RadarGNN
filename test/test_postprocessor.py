import numpy as np
import torch

from radargnn.utils.math import get_stats_of_predicted_box_points, point_iou, get_discrete_iou, is_point_in_rect
from detectron2.layers import nms_rotated


def test_nms_rotated():

    # every box in required format x, y, l, w, theta_x (degree)
    box_matrix = torch.tensor([[1, 2, 1, 1, 90],
                               [1, 2.9, 1, 1, 90]], dtype=torch.float64)

    scores = torch.tensor([0.2, 0.7], dtype=torch.float64)

    box_inters = 0.1 * 1
    box_union = (1 + 1) - (0.1 * 1)
    boxes_iou = box_inters / box_union

    # box iou is bigger than threshold
    # -> box with lower score is removed because boxes are too similar (box iou higher than allowed)
    iou_thresh_lower = boxes_iou - 0.01
    keep_lower = nms_rotated(
        box_matrix, scores, iou_thresh_lower).detach().numpy()
    res_1 = (keep_lower == np.array([1])).all()

    # box iou is lower than threshold
    # -> NO box will be removed -> boxes are too different
    iou_thresh_higher = boxes_iou + 0.01
    keep_higher = nms_rotated(
        box_matrix, scores, iou_thresh_higher).detach().numpy()
    res_2 = (keep_higher == np.array([1, 0])).all()

    assert ((res_1 and res_2))


def test_point_in_rectangle():

    box = np.array([[1, 1], [2, 1], [2, 0], [1, 0]])
    points = np.array([[1, 2], [2, 2], [0.5, 1], [1, 0.5], [
                      1.5, 0.5], [2, 0.5], [1.5, 0], [1.7, -0.001]])
    res_list = [False, False, False, True, True, True, True, False]

    pred_list = []
    for point in points:
        pred_list.append(is_point_in_rect(box, point))

    assert (res_list == pred_list)


def test_point_iou_rotated_box():
    boxes_pred = torch.tensor([[1, 1, 1, 1, 90],
                               [4, 4, 2, 2, 45]], dtype=torch.float64)

    boxes_gt = torch.tensor([[1, 0.9, 1, 1, 90],
                             [4.2, 3.9, 3, 2, 30]], dtype=torch.float64)

    points = np.array([[1, 2], [2, 3], [1, 1], [1, 1.45], [4, 4], [5, 4]])
    box_aligned = False

    iou_matrix_calc = point_iou(boxes_pred, boxes_gt, points, box_aligned)

    iou_matrix_true = torch.tensor([[0.5, 0],
                                    [0, 1]])

    assert ((iou_matrix_calc == iou_matrix_true).all().item())


def test_point_iou_algined_box():

    box_pred = torch.tensor([[1, 1, 2, 2], [2, 2, 3, 3]])
    box_true = torch.tensor([[3, 3, 4, 4], [1, 1, 2, 2], [5, 5, 8, 8]])

    points = np.array([[1, 1], [1.5, 1.5],
                       [2.5, 2.5],
                       [3.5, 3.5],
                       [6, 6], [7, 7]])

    aligned = True
    iou_matrix_calc = point_iou(box_pred, box_true, points, aligned)

    iou_matrix_true = torch.tensor([[0, 1, 0],
                                    [0, 0, 0]])
    assert ((iou_matrix_calc == iou_matrix_true).all().item())


def test_get_stats_of_predicted_box_points():

    p_pred = np.array([[1, 2], [2, 3], [-1, 7]])
    p_true = np.array([[1, 2], [-1, 7], [5, 6], [3, 2]])

    tp, fp, fn = get_stats_of_predicted_box_points(p_pred, p_true)

    assert (tp == 2 and fp == 1 and fn == 2)


def test_get_discrete_iou():
    iou = get_discrete_iou(2, 1, 2)
    assert (iou == 2 / 5)
