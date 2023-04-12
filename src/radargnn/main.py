
import argparse
import os

from radargnn.create_dataset import main as create_dataset
from radargnn.train import main as train
from radargnn.evaluate import main as evaluate

from radargnn.gnn.trainer import get_new_result_folder_path


def main(path_to_dataset: str, path_to_results: str, path_to_config: str):
    """
    Arguments:
        path_to_dataset: Path to the folder containing the raw dataset.
        path_to_results: Folder path to store the trained model.
        path_to_config: Path to the configuration file.
    """
    # path to the folder containing the processed dataset
    path_to_graph_data = f"{path_to_dataset}/processed"

    # get new result folder path, in which the results of "train()" are saved
    path_to_model_folder = get_new_result_folder_path(path_to_results)

    # create dataset - and store in ".../RadarScenes/processed" if not already present
    print("\n============ CREATING GRAPH DATASET =============\n")
    create_dataset(path_to_dataset, path_to_config)

    # train model on created or existing graph dataset ".../RadarScenes/processed"
    print("\n================ TRAINING MODEL =================\n")
    train(path_to_graph_data, path_to_results, path_to_config)

    # evaluate new trained model
    print("\n================ EVALUATING MODEL ===============\n")
    evaluate(path_to_graph_data, path_to_model_folder, path_to_config)


if __name__ == "__main__":
    # standard arguments
    file_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = f"{file_path[:-27]}data/datasets/nuscenes"
    results_path = f"{file_path[:-27]}data/results"
    config_path = f"{file_path[:-27]}configurations/configuration_nuscenes.yml"

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset_path)
    parser.add_argument('--results', type=str, default=results_path)
    parser.add_argument('--config', type=str, default=config_path)
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        raise Exception("Invalid path for dataset folder")

    if not os.path.isdir(args.results):
        raise Exception("Invalid path for results folder")

    if not os.path.isfile(args.config):
        raise Exception("Invalid path to config file")

    main(args.dataset, args.results, args.config)
