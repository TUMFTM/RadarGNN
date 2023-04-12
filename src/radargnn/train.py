
import argparse
import os
import torch

from radargnn.gnn.gnn_models import DetNetBasic
from radargnn.gnn.trainer import Trainer, set_seeds
from radargnn.utils.data_handling import get_data_loaders
from radargnn.utils.user_config_reader import UserConfigurationReader


def main(path_to_graph_data: str, path_to_results: str, path_to_config: str):
    """
    Arguments:
        path_to_graph_data: Path to the folder containing the processed dataset.
        path_to_results: Folder path to store the trained model.
        path_to_config: Path to the configuration file.
    """

    # create config objects
    config_dict = UserConfigurationReader.read_config_file(path_to_config)
    model_config = UserConfigurationReader.get_config_object("MODEL_ARCHITECTURE", config_dict)
    training_config = UserConfigurationReader.get_config_object("TRAINING", config_dict)

    # check device to train on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Train device: {device}")

    # fix seed to maximize determinism if desired
    if training_config.deterministic:
        set_seeds(training_config.seed)

    # data loader creation
    print(">>> loading data")
    data_loaders, dataset_config_dict = get_data_loaders(
        ["train", "validate"], path_to_graph_data, training_config.batch_size, training_config.shuffle)

    # create gnn
    print(">>> creating model")
    model = DetNetBasic(model_config)

    # train model
    print(">>> starting training")
    trainer = Trainer(training_config, model)
    trainer.fit(data_loaders)

    # save model, its configs and losses to new folder in results folder
    trainer.save_results(path_to_results, model_config, dataset_config_dict)


if __name__ == "__main__":

    # standard arguments
    file_path = os.path.dirname(os.path.realpath(__file__))
    graph_data_path = f"{file_path[:-27]}data/datasets/nuscenes/processed"
    results_path = f"{file_path[:-27]}data/results"
    config_path = f"{file_path[:-27]}configurations/configuration_nuscenes.yml"

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=graph_data_path)
    parser.add_argument('--results', type=str, default=results_path)
    parser.add_argument('--config', type=str, default=config_path)
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        raise Exception("Invalid path for graph data folder")

    if not os.path.isdir(args.results):
        raise Exception("Invalid path for results folder")

    if not os.path.isfile(args.config):
        raise Exception("Invalid path to config file")

    main(args.data, args.results, args.config)
