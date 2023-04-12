from typing import Dict

from radargnn.postprocessor.evaluation import Evaluator
from radargnn.postprocessor.nuscenes.evaluation import NuscenesEvaluator
from radargnn.postprocessor.radarscenes.evaluation import RadarscenesEvaluator


evaluation_selector: Dict[str, Evaluator] = {
    "radarscenes": RadarscenesEvaluator,
    "nuscenes": NuscenesEvaluator
}
