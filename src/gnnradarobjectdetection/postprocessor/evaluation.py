import abc
import glob
from typing import Dict, List

from gnnradarobjectdetection.postprocessor.configs import PostProcessingConfiguration


class Evaluator(abc.ABC):
    def __init__(self, config: PostProcessingConfiguration, *args, **kwargs):
        self.config = config
        self.names = list(self.config.min_object_score.keys())
        self.names.insert(self.config.bg_index, "background")

    @abc.abstractmethod
    def evaluate(self, predictions: Dict[str, List], ground_truth: Dict[str, Dict], num_predictions: int, pos: List) -> None:
        pass

    @abc.abstractmethod
    def save_results(self, path_to_model_folder: str) -> None:
        pass


def get_new_evaluation_folder_path(path: str) -> str:
    """ Returns the path to store the evaluation results.

    One model may be evaluated multiple times.
    Each evaluation result is stored in a folder with increasing index number (evaluation_01, evaluation_02, ...).
    This function returns the path to the evaluation folder with the next higher index number.

    Args:
        path: Path to the parent folder in which the model and all its evaluation folders are stored.

    Returns:
        folder_path: Path to the evaluation folder with the next higher index number.
    """
    folders = glob.glob(path + "/*/")

    if len(folders) == 0:
        folder_name = "evaluation_01"

    else:
        numbers = []
        for folder in folders:
            splits = folder.split("_")
            number = splits[-1][:-1]
            try:
                numbers.append(int(number))
            except ValueError:
                pass

        next_number = max(numbers) + 1
        if next_number < 10:
            next_number_str = f"0{str(next_number)}"
        else:
            next_number_str = str(next_number)

        folder_name = f"evaluation_{next_number_str}"

    folder_path = f"{path}/{folder_name}"

    return folder_path
