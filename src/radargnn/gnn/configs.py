from dataclasses import dataclass, field


@dataclass
class GNNArchitectureConfig:
    """ Stores possible GNN model architecture configurations.
    """

    # initial node and edge feature dimension
    node_feature_dimension: int
    edge_feature_dimension: int

    # layers for graph convolution and detection head
    conv_layer_dimensions: list
    classification_head_layer_dimensions: list
    regression_head_layer_dimensions: list

    # layers for initial node and edge feature embedding MLPs
    initial_node_feature_embedding: bool = False
    initial_edge_feature_embedding: bool = False
    node_feature_embedding_layer_dimensions: list = None
    edge_feature_embedding_layer_dimensions: list = None
    conv_layer_type: str = "MPNNConv"

    # configuration for graph convolution layers
    batch_norm_in_mlps: bool = True
    conv_pre_mlp_layer_number: int = 1
    conv_post_mlp_layer_number: int = 1
    conv_use_edge_encoder: bool = False
    aggregation_function: str = "max"


@dataclass
class TrainingConfig:
    """ Stores training hyperparameter of model.
    """
    dataset: str

    learning_rate: float
    epochs: int
    batch_size: int
    shuffle: bool

    bg_index: int

    deterministic: bool = False
    seed: int = 0

    # weights of every class for negative log likelihood loss in classification
    class_weights: dict = field(default_factory=dict)
    set_weights_according_radar_scenes_distribution: bool = False

    # validation class weights
    val_class_weights: dict = field(default_factory=dict)

    bb_loss_weight: float = 1
    cls_loss_weight: float = 1

    regularization_strength: float = 1e-4
    reduce_lr_on_plateau_factor: float = 0.5
    reduce_lr_on_plateau_patience: int = 0
    exponential_lr_decay_factor: float = 0.0

    early_stopping_patience: int = 10

    # configuration to adapt orientation angle representation
    adapt_orientation_angle: bool = False

    def __post_init__(self):
        # use default class weights if not specified otherwise
        if self.dataset == "radarscenes":
            self.class_weights.setdefault('car', 1)
            self.class_weights.setdefault('pedestrian', 1)
            self.class_weights.setdefault('pedestrian_group', 1)
            self.class_weights.setdefault('two_wheeler', 1)
            self.class_weights.setdefault('large_vehicle', 1)
            self.class_weights.setdefault('background', 0.05)

        elif self.dataset == "nuscenes":
            self.class_weights.setdefault('background', 0.05)
            self.class_weights.setdefault('barrier', 1)
            self.class_weights.setdefault('bicycle', 1)
            self.class_weights.setdefault('bus', 1)
            self.class_weights.setdefault('car', 1)
            self.class_weights.setdefault('construction', 1)
            self.class_weights.setdefault('motorcycle', 1)
            self.class_weights.setdefault('pedestrian', 1)
            self.class_weights.setdefault('trafficcone', 1)
            self.class_weights.setdefault('trailer', 1)
            self.class_weights.setdefault('truck', 1)

        else:
            raise ValueError("Only the radarscenes and nuscenes dataset are supported!")

        if self.val_class_weights:
            # Check if the class weights and validation class weights have the same classes
            assert set(self.class_weights.keys()) == set(self.val_class_weights.keys())
        else:
            # if not specified use the training class weights for validation
            self.val_class_weights = self.class_weights
