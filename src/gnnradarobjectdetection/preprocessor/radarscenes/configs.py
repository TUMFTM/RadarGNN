from dataclasses import dataclass
from radar_scenes.sequence import get_training_sequences, get_validation_sequences


@dataclass()
class RadarScenesDatasetConfiguration:
    """ Stores settings for creating point clouds from the RadarScenes dataset.
    """
    time_per_point_cloud_frame: float
    crop_point_cloud: bool
    crop_settings: dict
    bounding_boxes_aligned: bool
    bb_invariance: str
    create_small_subset: bool
    subset_settings: dict = None

    deterministic: bool = False
    seed: int = 0

    parallelize: bool = False


@dataclass
class RadarScenesSplitConfiguration:
    """ Stores sequences used for creating the training, testing, validation split of the dataset.

    TODO: Outsource specification of test sequences (idx_test) into a separate file.
    """
    sequence_dict: dict

    def __init__(self, sequence_file: str, standard_split: bool = True, train_sequences: list = [],
                 test_sequences: list = [], validate_sequences: list = []):

        if standard_split:
            # use standard radar scenes split into train and validate and additional here defined test split
            sequence_list_train_test = get_training_sequences(sequence_file)

            # indexes for training and testing dataset from train_test
            all_idx = {i for i in range(len(sequence_list_train_test))}

            # define sequences to be used for testing, remaining sequences used for training
            idx_test = {4, 6, 11, 16, 18, 24, 33, 34, 36, 37, 42, 44, 48, 52,
                        53, 60, 63, 67, 73, 84, 86, 92, 94, 100, 108, 119, 124, 126}
            idx_train = all_idx - idx_test

            # split into ca: 64% train, 18% test, 18% validate
            # (split in Scheiner et. al.: 64% train, 20% test, 16% validate)
            sequence_list_train = [sequence_list_train_test[i] for i in idx_train]
            sequence_list_test = [sequence_list_train_test[i] for i in idx_test]
            sequence_list_validate = get_validation_sequences(sequence_file)

        else:
            sequence_list_train = train_sequences
            sequence_list_test = test_sequences
            sequence_list_validate = validate_sequences

        sequence_dict = {"train": sequence_list_train,
                         "test": sequence_list_test,
                         "validate": sequence_list_validate}

        self.sequence_dict = sequence_dict
