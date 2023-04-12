import json
import glob
from torch_geometric.loader import DataLoader
import torch


def get_data_loaders(splits: list, root: str, batch_size: int, shuffle: bool) -> dict:
    """
    Reads graphs and stores them in pytorch dataloaders for training and validation.

    Args:
        splits: list of strings for which split a dataloader should be created
        root: string, path to the processed graph dataset
        batch_size: integer, batch size in the dataloader
        shuffle: bool if the graphs should be shuffled or stay in order in the dataloader

    Returns:
        data_loaders: dict containing dataloaders as values with its split names as keys
     """
    data_loaders = {}
    for split in splits:
        graph_list = []
        path_to_split = f"{root}/{split}"

        graph_names = sorted(glob.glob(f"{path_to_split}/*.pt"))
        for graph_name in graph_names:
            graph = torch.load(graph_name)
            graph_list.append(graph)

        data_loaders[split] = DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)

    path_to_config = f"{root}/config.json"
    with open(path_to_config, 'r') as f:
        dataset_config_dict = json.load(f)

    return data_loaders, dataset_config_dict
