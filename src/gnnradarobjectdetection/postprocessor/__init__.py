from typing import Dict

from gnnradarobjectdetection.postprocessor.evaluation import Evaluator
from gnnradarobjectdetection.postprocessor.nuscenes.evaluation import NuscenesEvaluator
from gnnradarobjectdetection.postprocessor.radarscenes.evaluation import RadarscenesEvaluator


evaluation_selector: Dict[str, Evaluator] = {
    "radarscenes": RadarscenesEvaluator,
    "nuscenes": NuscenesEvaluator
}
