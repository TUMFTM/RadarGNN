from dataclasses import dataclass


@dataclass
class GraphConstructionConfiguration:
    """ Stores all settings required for creating a graph based on the point cloud
    """

    graph_construction_algorithm: str
    graph_construction_settings: dict

    node_features: list
    edge_features: list
    edge_mode: str

    distance_definition: str

    def __post_init__(self):
        if self.graph_construction_algorithm == "knn":
            self.k = self.graph_construction_settings.get("k")
            self.r = None
        elif self.graph_construction_algorithm == "radius":
            self.r = self.graph_construction_settings.get("r")
            self.k = None
        else:
            raise Exception("Invalid graph construction algorithm selected")
