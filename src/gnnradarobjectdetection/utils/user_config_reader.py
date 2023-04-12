import yaml
import dataclasses

from gnnradarobjectdetection import preprocessor
from gnnradarobjectdetection.postprocessor.configs import PostProcessingConfiguration
from gnnradarobjectdetection.preprocessor.configs import GraphConstructionConfiguration
from gnnradarobjectdetection.gnn.configs import GNNArchitectureConfig, TrainingConfig


def dataclass_from_dict(data_class, d):
    """ Converts a dict into a dataclass object.
    """
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(data_class)}
        return data_class(**{f: dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except Exception:
        return d


class ConfigToDataClassMapping:

    @staticmethod
    def get_mapping_dicts(dataset: str):

        dataclass_mapping_dict = {"DATASET_PROCESSING": preprocessor.config_selector[dataset],
                                  "GRAPH_CONSTRUCTION": GraphConstructionConfiguration,
                                  "MODEL_ARCHITECTURE": GNNArchitectureConfig,
                                  "TRAINING": TrainingConfig,
                                  "POSTPROCESSING": PostProcessingConfiguration}

        supertask_mapping_dict = {"DATASET_PROCESSING": "CREATE_DATASET",
                                  "GRAPH_CONSTRUCTION": "CREATE_DATASET",
                                  "MODEL_ARCHITECTURE": "TRAIN",
                                  "TRAINING": "TRAIN",
                                  "POSTPROCESSING": "EVALUATE"}

        return dataclass_mapping_dict, supertask_mapping_dict


class UserConfigurationReader():

    @staticmethod
    def get_config_object(config_subset_name: str, config_dict: dict):
        """ Transforms configuration.yml file into the corresponding dataclass instance.
        """
        dataset = config_dict['CREATE_DATASET']['dataset']

        dataclass_mapping_dict, supertask_mapping_dict = ConfigToDataClassMapping.get_mapping_dicts(dataset)

        get_super_task = supertask_mapping_dict.get(config_subset_name)
        subset_config_dict = config_dict.get(
            get_super_task).get(config_subset_name)

        config = dataclass_from_dict(dataclass_mapping_dict.get(
            config_subset_name), subset_config_dict)

        if not isinstance(config, dataclass_mapping_dict.get(config_subset_name)):
            raise Exception("Conversion of config file to dataclass failed.")

        return config

    @staticmethod
    def read_config_file(path: str):
        with open(path) as f:
            config = yaml.safe_load(f)
        return config
