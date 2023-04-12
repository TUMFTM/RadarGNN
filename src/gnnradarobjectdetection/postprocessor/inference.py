import torch
from torch.nn import Softmax


class Predictor():
    """
    Makes predictions for whole evaluation dataset and stores raw prediction results.

    Properties:
        model: Trained PyTorch model.
        dataloader: PyTorch data loader.
    """

    def __init__(self, model, dataloader, verbose: bool = True):
        self.model = model
        self.dataloader = dataloader
        self.verbose = verbose

    def predict(self):
        """Predicts output for each graph in the Dataloader.

        Applies Softmax function to the GNNs' raw class predictions to obtain a probability distribution over the n classes.
        Extracts ground truth from graph data.

        Returns:
            predictions: Dict with bounding box and class label predictions of each node of each graph of the Dataloader.
            ground_truth: Dict with ground truth bounding box and class label of each node of each graph of the Dataloader.
            pos: Contains the spatial coordinates of each node of each graph in the Dataloader.
            vel: Contains the velocity vector of each node of each graph in the Dataloader.
        """
        # Initialize return values
        pos = []
        vel = []

        predictions = {}
        predictions["bounding_box_predictions"] = []
        predictions["class_probability_prediction"] = []

        ground_truth = {}
        ground_truth["bounding_box_true"] = []
        ground_truth["class_true"] = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device for inference: {device}")
        self.model.to(device)
        softmax_func = Softmax(dim=1).to(device)

        for i, graph_batch in enumerate(self.dataloader):

            # extract and store ground truth and positions
            pos.append(graph_batch.pos.detach().numpy())
            vel.append(graph_batch.vel.detach().numpy())
            ground_truth["class_true"].append(graph_batch.y[:, 0].detach().numpy())
            ground_truth["bounding_box_true"].append(graph_batch.y[:, 1:].detach().numpy())

            # make predictions
            graph_batch.to(device)
            cls, bb = self.model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)

            # apply Softmax to get probability prediction for classes
            # (because this is not done in detection head of gnn)
            cls_prob = softmax_func(cls)

            # get predictions back to cpu and store results
            bb = bb.cpu()
            cls_prob = cls_prob.cpu()
            predictions["bounding_box_predictions"].append(bb.detach().numpy())
            predictions["class_probability_prediction"].append(cls_prob.detach().numpy())

            # print progress
            if self.verbose:
                if (i + 1) == 1 or (i + 1) % 10 == 0 or (i + 1) == (len(self.dataloader)):
                    print(f"{i+1}/{len(self.dataloader)} inferences finished")

        return predictions, ground_truth, pos, vel
