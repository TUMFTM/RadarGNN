from radargnn.preprocessor.radarscenes.dataset_creation import RadarScenesGraphDataset
from radargnn.preprocessor.nuscenes.dataset_creation import NuScenesGraphDataset

from radargnn.preprocessor.radarscenes.configs import RadarScenesDatasetConfiguration
from radargnn.preprocessor.nuscenes.configs import NuScenesDatasetConfiguration

dataset_selector = {
    'radarscenes': RadarScenesGraphDataset,
    'nuscenes': NuScenesGraphDataset
}

config_selector = {
    'radarscenes': RadarScenesDatasetConfiguration,
    'nuscenes': NuScenesDatasetConfiguration
}
