import argparse
import glob
import os
import torch

from radargnn import postprocessor
from radargnn.postprocessor.inference import Predictor
from radargnn.postprocessor.postprocessing import Postprocessor, PredictionExtractor
from radargnn.utils.data_handling import get_data_loaders
from radargnn.utils.user_config_reader import UserConfigurationReader


def main(path_to_data: str, path_to_model_folder: str, path_to_config: str):
    """ Evaluates the trained model on the specified dataset.

    Arguments:
        path_to_data: Path to the folder containing the dataset (raw and processed).
        path_to_model_folder: Path to the folder containing the trained model.
        path_to_config: Path to the configuration file.
    """
    # get data paths
    path_to_raw_data = os.path.join(path_to_data, "raw")
    path_to_graph_data = os.path.join(path_to_data, "processed")

    # create config objects
    config_dict = UserConfigurationReader.read_config_file(path_to_config)
    config = UserConfigurationReader.get_config_object("POSTPROCESSING", config_dict)

    # get selected dataset information
    dataset = config_dict['CREATE_DATASET'].get('dataset', None)
    version = config_dict['CREATE_DATASET']['DATASET_PROCESSING'].get('version', None)

    # get sample file names
    graph_names = sorted(glob.glob(f"{path_to_graph_data}/{config.split}/*.pt"))

    # get data loader for evaluation
    splits = [config.split]

    # evaluation graph by graph (batchsize = 1) without shuffling
    eval_loader_dict, _ = get_data_loaders(splits, path_to_graph_data, batch_size=1, shuffle=False)
    eval_loader = eval_loader_dict.get(splits[0])

    # remove eval loader dict from storage
    eval_loader_dict = None

    # load model (try first loading model directly)
    # otherwise create model based on model_config and then load state dict into model
    try:
        model = torch.load(f"{path_to_model_folder}/trained_model.pt", map_location=torch.device('cpu'))
    except Exception:
        # TODO: Create model based on configs and initialize with loaded state dict
        raise Exception("Loading the trained model failed")

    # make predictions for whole dataloader
    predictor = Predictor(model, eval_loader)
    predictions, ground_truth, pos, vel = predictor.predict()

    # delete predictor, model and eval_loader
    del predictor, model, eval_loader

    # postprocess predictions
    post_processor = Postprocessor()
    bb_pred, bb_ground_truth, cls_pred, cls_ground_truth = post_processor.process(config, pos, vel, predictions, ground_truth)

    prediction_extractor = PredictionExtractor()
    cls_pred_label = prediction_extractor.extract(predictions)

    # delete raw data (predictions and ground truth)
    del predictions, ground_truth, pos

    # Select and initialize evaluator
    Evaluator = postprocessor.evaluation_selector[dataset]
    eval = Evaluator(config=config, version=version, dataset_path=path_to_raw_data, model_path=path_to_model_folder)

    # postprocess and evaluate raw predictions
    eval.evaluate(bb_pred, bb_ground_truth, cls_pred, cls_pred_label, cls_ground_truth, vel, graph_names=graph_names)
    eval.save_results(path_to_model_folder)


if __name__ == "__main__":
    # standard arguments
    file_path = os.path.dirname(os.path.realpath(__file__))

    # Should of course be the same dataset as used for training
    data_path = f"{file_path[:-27]}data/datasets/nuscenes"
    model_folder_path = f"{file_path[:-27]}data/results/model_01"
    config_path = f"{file_path[:-27]}configurations/configuration_nuscenes.yml"

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=data_path)
    parser.add_argument('--model', type=str, default=model_folder_path)
    parser.add_argument('--config', type=str, default=config_path)
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        raise Exception("Invalid path for graph data folder")

    if not os.path.isdir(args.model):
        raise Exception("Invalid path for model folder")

    if not os.path.isfile(args.config):
        raise Exception("Invalid path to config file")

    main(args.data, args.model, args.config)
