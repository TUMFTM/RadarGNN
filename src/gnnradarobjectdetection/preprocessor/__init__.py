from gnnradarobjectdetection.preprocessor.radarscenes.dataset_creation import RadarScenesGraphDataset
from gnnradarobjectdetection.preprocessor.nuscenes.dataset_creation import NuScenesGraphDataset

from gnnradarobjectdetection.preprocessor.radarscenes.configs import RadarScenesDatasetConfiguration
from gnnradarobjectdetection.preprocessor.nuscenes.configs import NuScenesDatasetConfiguration

dataset_selector = {
    'radarscenes': RadarScenesGraphDataset,
    'nuscenes': NuScenesGraphDataset
}

config_selector = {
    'radarscenes': RadarScenesDatasetConfiguration,
    'nuscenes': NuScenesDatasetConfiguration
}
