from dataclasses import dataclass

from radargnn.preprocessor.nuscenes import splits


@dataclass()
class NuScenesDatasetConfiguration:
    """
    Stores further dataset creation deteails
    """
    version: str = 'v1.0-trainval'
    nsweeps: int = 1
    crop_point_cloud: bool = False
    crop_settings: dict = None
    wlh_factor: float = 1.0
    wlh_offset: float = 0.0
    bounding_boxes_aligned: bool = False
    bb_invariance: str = "translation"
    deterministic: bool = False
    seed: int = 0


@dataclass
class NuScenesSplitConfiguration:
    """
    Stores sequences used for creating the training, testing, validation split of the dataset.
    """
    sequence_dict: dict

    def __init__(self, version: str = 'v1.0-mini'):
        if version == 'v1.0-mini':
            self.sequence_dict = {
                'train': list(sorted(set(splits.mini_train))),
                'validate': list(sorted(set(splits.mini_val)))
            }

        elif version == 'v1.0-trainval':
            self.sequence_dict = {
                'train': list(sorted(set(splits.train_detect + splits.train_track))),
                'validate': list(sorted(set(splits.val)))
            }
        elif version == 'v1.0-test':
            self.sequence_dict = {
                'test': list(sorted(set(splits.test)))
            }
        else:
            raise ValueError(f'The given dataset version {version} is not'
                             'a valid version of the nuScenes Dataset!')
