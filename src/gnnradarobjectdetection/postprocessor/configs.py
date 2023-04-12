from dataclasses import dataclass


@dataclass
class PostProcessingConfiguration:
    """ Holds configurations for postprocessing and evaluation.
    """

    # split to evaluate
    split: str

    iou_for_nms: float
    min_object_score: dict
    max_score_for_background: float

    iou_for_mAP: float = 0.3
    use_point_iou: bool = False

    bg_index: int = 5

    bb_invariance: str = "translation"
    adapt_orientation_angle: bool = False

    get_mAP: bool = True
    get_confusion: bool = True
    get_segmentation_f1: bool = True
    f1_class_averaging: str = None
