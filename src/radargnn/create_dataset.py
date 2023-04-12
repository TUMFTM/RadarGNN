import os
import argparse

from radargnn import preprocessor
from radargnn.utils.user_config_reader import UserConfigurationReader
from radargnn.gnn.trainer import set_seeds


def main(path_to_dataset: str, path_to_config: str):
    """ Creates the desired dataset for model training and evaluation.

    Arguments:
        path_to_dataset: Path to the folder containing the raw dataset.
        path_to_config: Path to the configuration file.
    """

    # create configuration objects
    config_dict = UserConfigurationReader.read_config_file(path_to_config)
    dataset_config = UserConfigurationReader.get_config_object("DATASET_PROCESSING", config_dict)
    graph_config = UserConfigurationReader.get_config_object("GRAPH_CONSTRUCTION", config_dict)

    # fix seed to maximize determinism if desired
    if dataset_config.deterministic:
        set_seeds(dataset_config.seed)

    # get selected dataset
    dataset = config_dict['CREATE_DATASET']['dataset']

    # create graph dataset for training, testing, validation and save it
    GraphDataset = preprocessor.dataset_selector[dataset]
    GraphDataset(path_to_dataset, graph_config, dataset_config)


if __name__ == "__main__":

    # standard arguments
    file_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = f"{file_path[:-27]}data/datasets/nuscenes"
    config_path = f"{file_path[:-27]}configurations/configuration_nuscenes.yml"

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset_path)
    parser.add_argument('--config', type=str, default=config_path)
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        raise Exception("Invalid path for dataset folder")

    if not os.path.isfile(args.config):
        raise Exception("Invalid path to config file")

    main(args.dataset, args.config)
