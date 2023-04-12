
import os
import glob
import json
import torch
import time
import copy
from dataclasses import asdict
import torch_geometric

from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, LambdaLR
import matplotlib.pyplot as plt
import numpy as np

from radargnn.gnn.configs import TrainingConfig, GNNArchitectureConfig
from radargnn.utils.radar_scenes_properties import ClassDistribution
from radargnn.preprocessor.bounding_box import adapt_bb_orientation_angle


class Trainer():
    """ Trainer that can be used for GNN training, visualization of training results and saving the results.

    Attributes:
        config: Configuration dataclass object.
        model: Graph neural network model.
        train_loss_cls: List that stores the classification loss during training.
        train_loss_bb: List that stores the bounding box regression loss during training.
        train_loss: List that stores the overall loss during training.
        valid_loss: List that stores the overall loss during validation.
        model_lowest_valid:

    Methods:
        fit: Execute complete model training.
        save_results: Save results of the model training.
        show_learning_curves: Visualize learning curves.
    """

    def __init__(self, config: TrainingConfig, model: torch.nn.Module):
        """
        Args:
            config: TrainingConfig object that contains the configuration information.
            model: torch.nn.Module object in form of the GNN for object detection.
        """

        self.config = config
        self.model = model
        self.train_loss_cls = []
        self.train_loss_bb = []
        self.train_loss = []
        self.valid_loss = []
        self.model_lowest_valid = {}

    def fit(self, data_loaders: dict) -> None:
        """ Train the model.

        This method trains the model with the dataset and therefore updates the models parameters.
        For this, it uses the configuration that was passed during the creation of the trainer instance.

        Args:
            data_loaders: Dict of pytorch data loaders containing training, test and validation data.
        """

        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        lr = self.config.learning_rate
        regularization_strength = self.config.regularization_strength
        # weight decay is in this implementation equivalent to L2 regularization
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=regularization_strength)

        # Determine learning rate scheduler
        if self.config.reduce_lr_on_plateau_patience > 0:
            # Apply learning rate decay on plateau after patience epochs without change
            reduce_lr_on_plateau_factor = self.config.reduce_lr_on_plateau_factor
            reduce_lr_on_plateau_patience = self.config.reduce_lr_on_plateau_patience
            scheduler = ReduceLROnPlateau(optimizer, factor=reduce_lr_on_plateau_factor, patience=reduce_lr_on_plateau_patience)

        elif self.config.exponential_lr_decay_factor > 0:
            # Apply exponential learning rate decay every epoch (lr * gamma ** epoch)
            gamma = self.config.exponential_lr_decay_factor
            scheduler = ExponentialLR(optimizer, gamma=gamma)

        else:
            # Apply no learning rate decay (use constant initial learning rate)
            scheduler = LambdaLR(optimizer, lambda _: 1.0)

        # define weights for different classes in cross entropy loss (for unbalanced dataset)
        if self.config.set_weights_according_radar_scenes_distribution:
            class_weight_dict = ClassDistribution.get_class_weights()
            weights = torch.tensor(list(class_weight_dict.values()), dtype=torch.float32)
            val_weights = weights
        else:
            weights = torch.tensor(list(self.config.class_weights.values()), dtype=torch.float32)
            val_weights = torch.tensor(list(self.config.val_class_weights.values()), dtype=torch.float32)

        # applies torch.nn.LogSoftmax followed by torch.nn.NLLLoss
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)
        cross_entropy.to(device)

        val_cross_entropy = torch.nn.CrossEntropyLoss(weight=val_weights)
        val_cross_entropy.to(device)

        # define huber loss for bounding boxes
        huber = torch.nn.HuberLoss()

        # train for epochs
        num_epochs = self.config.epochs
        early_topping_triggers = 0
        for epoch in range(1, num_epochs + 1):

            loss_train, loss_cls, loss_bb = self.__train(data_loaders.get("train"), device, optimizer, cross_entropy, huber)
            loss_valid = self.__test(data_loaders.get("validate"), device, val_cross_entropy, huber)

            self.train_loss.append(loss_train)
            self.train_loss_cls.append(loss_cls)
            self.train_loss_bb.append(loss_bb)
            self.valid_loss.append(loss_valid)

            # Update learning rate according to the lr scheduler
            if self.config.reduce_lr_on_plateau_patience > 0:
                # pass validation loss to scheduler
                scheduler.step(loss_valid)
            else:
                scheduler.step()

            # store model with lowest validation loss
            if loss_valid <= min(self.valid_loss):
                self.model_lowest_valid = {"model": copy.deepcopy(self.model), "epoch": epoch}

            # print loss
            if epoch == 1 or epoch % 1 == 0 or epoch == num_epochs:
                print(f">>> Epoch: {epoch}/{num_epochs}, loss_train: {round(loss_train,5)}, loss_valid: {round(loss_valid,5)}")

            # Early stopping
            if loss_valid > min(self.valid_loss):
                early_topping_triggers += 1
                print('Trigger Times:', early_topping_triggers)

                if early_topping_triggers >= self.config.early_stopping_patience:
                    # Stopping if no new validation loss minimum is reached x times in a row
                    print('Early stopping!')
                    break
            else:
                early_topping_triggers = 0

        end_time = time.time()
        print(
            f">>> Overall training duration: {round(((end_time - start_time)/(60*60)), 2)} hours")

    def __train(self, train_dataloader: DataLoader, device, optimizer, cross_entropy, huber) -> float:
        """ Executes one training epoch with the given dataloader.

        Executes a forward pass, calculates the loss, and updates the model parameters using all available training data.

        """
        loss_epoch = 0
        loss_cls_epoch = 0
        loss_bb_epoch = 0
        loss_bb_nan_count = 0

        # iterate over training data loader batches
        for idx, graph_batch in enumerate(train_dataloader):

            # change angle representation of bounding box (currently in rad from 0-pi measured from x-axis to long bb side)
            if self.config.adapt_orientation_angle:
                bb_matrix_adapt = adapt_bb_orientation_angle(graph_batch.y[:, 1:])
                graph_batch.y = torch.cat((graph_batch.y[:, :1], bb_matrix_adapt[:, :]), dim=1)

            start_time = time.time()
            # add graph_batch to device
            graph_batch.to(device)

            # remove gradients
            optimizer.zero_grad()

            # forward pass
            graph_batch.x.requires_grad_()
            graph_batch.edge_attr.requires_grad_()
            cls, bb = self.model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)

            # get ground truth
            label_true = graph_batch.y[:, 0].long()
            bb_true = graph_batch.y[:, 1:]

            # loss for classification -> L_cls
            loss_cls = cross_entropy(cls, label_true)

            # if point is no background -> Calculate bounding box loss
            loss_bb = 0
            num_bb = 0
            for i, label in enumerate(label_true):
                if label != self.config.bg_index:
                    num_bb += 1

                    current_bb_true = bb_true[i, :]
                    current_bb_pred = bb[i, :]

                    loss_one_bb = huber(current_bb_true, current_bb_pred)
                    loss_bb += loss_one_bb

            if num_bb != 0:
                loss_bb = loss_bb / num_bb
            else:
                loss_bb = 0

            # if a nan bb loss is returned -> ignore loss of the batch to not terminate training
            # -> very rare bug for rotation invariant bounding boxes and point pair features
            # -> occurs in in 3 / 21.000 train graphs
            # -> reason for the bug is still unknown
            try:
                # only works if loss_bb is tensor and not zero anyway
                if np.isnan(loss_bb.item()):
                    loss_bb = 0
                    loss_bb_nan_count += 1
                    print(f">>> nan in loss_bb found and ignored for the {loss_bb_nan_count} time <<<")
            except Exception:
                # if los_bb is already zero due to no present bounding boxes in batch
                pass

            # sum up loss terms (regularization is added via adams weight_decay)
            alpha = self.config.cls_loss_weight  # weight for loss term balancing
            beta = self.config.bb_loss_weight  # weight for loss term balancing
            loss = alpha * loss_cls + beta * loss_bb

            # backpropagate
            loss.backward()

            # adapt weights
            optimizer.step()

            loss_epoch += loss.item()
            loss_cls_epoch += loss_cls.item()
            if loss_bb == 0:
                loss_bb_epoch += loss_bb
            else:
                loss_bb_epoch += loss_bb.item()

            end_time = time.time()
            print(f"Batch {idx+1}/{len(train_dataloader)}: required time - {round((end_time - start_time),2)} s")

        # divide loss by number of batches to be independent of that batch number
        # -> To compare loss of training and testing set with much less batched graphs in loader
        loss_epoch = loss_epoch / len(train_dataloader)
        loss_cls_epoch = loss_cls_epoch / len(train_dataloader)
        loss_bb_epoch = loss_bb_epoch / len(train_dataloader)

        return loss_epoch, loss_cls_epoch, loss_bb_epoch

    @torch.no_grad()
    def __test(self, test_dataloader: DataLoader, device, cross_entropy, huber) -> float:
        """ Executes one validation epoch with the given dataloader..

        Executes a forward pass and calculates the loss using all available validation data.
        Does NOT update the model parameters based on the calculated loss.

        """
        loss_epoch = 0

        # iterate over training data loader batches
        for i, graph_batch in enumerate(test_dataloader):

            # change angle representation of bounding box (currently in rad from 0-pi measured from x-axis to long bb side)
            if self.config.adapt_orientation_angle:
                bb_matrix_adapt = adapt_bb_orientation_angle(graph_batch.y[:, 1:])
                graph_batch.y = torch.cat((graph_batch.y[:, :1], bb_matrix_adapt[:, :]), dim=1)

            # add graph_batch to device
            graph_batch.to(device)

            # forward pass without gradient tracking
            cls, bb = self.model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)

            # get ground truth
            label_true = graph_batch.y[:, 0].long()
            bb_true = graph_batch.y[:, 1:]

            # loss for classification -> L_cls
            loss_cls = cross_entropy(cls, label_true)

            # if point is no background -> Calculate bounding box loss
            loss_bb = 0
            num_bb = 0
            for i, label in enumerate(label_true):
                if label != self.config.bg_index:
                    num_bb += 1

                    current_bb_true = bb_true[i, :]
                    current_bb_pred = bb[i, :]

                    loss_one_bb = huber(current_bb_true, current_bb_pred)
                    loss_bb += loss_one_bb

            if num_bb != 0:
                loss_bb = loss_bb / num_bb
            else:
                loss_bb = 0

            # sum up loss terms (regularization is added via adams weight_decay)
            alpha = self.config.cls_loss_weight  # weight for loss term balancing
            beta = self.config.bb_loss_weight  # weight for loss term balancing
            loss = alpha * loss_cls + beta * loss_bb
            loss_epoch += loss.item()

        # divide loss by number of batches to be independent of that batch number
        # -> To compare loss of training and testing set with much less batched graphs in loader
        loss_epoch = loss_epoch / len(test_dataloader)
        return loss_epoch

    def save_results(self, path: str, model_config: GNNArchitectureConfig, dataset_config_dict: dict) -> None:
        """ Saves the results of the training.

        Saves the trained model, the used configurations, and the loss data inside the specified folder.

        Args:
            path: Path to a folder in which the results are saved.
            model_config: Configuration of the model.
            dataset_config_dict: Configuration of the graph dataset creation.
        """

        folder_path = get_new_result_folder_path(path)
        os.mkdir(folder_path)

        # create dictionaries that can then be written into Json file
        gnn_config_dict = asdict(model_config)
        training_config_dict = asdict(self.config)

        json_dict = {"GNN_ARCHITECTURE_CONFIG": gnn_config_dict,
                     "TRAINING_CONFIG": training_config_dict}

        # write JSON file for model config
        json_path = f"{folder_path}/gnn_configs.json"
        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=4)

        # write JSON file for dataset config
        json_path = f"{folder_path}/dataset_configs.json"
        with open(json_path, 'w') as f:
            json.dump(dataset_config_dict, f, indent=4)

        # save model as "whole" and as state dict of parameters
        model_path = f"{folder_path}/trained_model.pt"
        torch.save(self.model, model_path)

        model_state_dict_path = f"{folder_path}/trained_model_state_dict.pt"
        torch.save(self.model.state_dict(), model_state_dict_path)

        # save model with lowest validation loss as "whole" and as state dict of parameters
        model_path = f"{folder_path}/trained_model_low_val_ep{self.model_lowest_valid.get('epoch')}.pt"
        torch.save(self.model_lowest_valid.get("model"), model_path)

        model_state_dict_path = f"{folder_path}/trained_model_low_val_ep{self.model_lowest_valid.get('epoch')}_state_dict.pt"
        torch.save(self.model_lowest_valid.get("model").state_dict(), model_state_dict_path)

        # save loss curves
        loss_train = np.array([self.train_loss])
        loss_validation = np.array([self.valid_loss])
        loss_train_cls = np.array([self.train_loss_cls])
        loss_train_bb = np.array([self.train_loss_bb])

        with open(f"{folder_path}/loss_train.npy", 'wb') as f:
            np.save(f, loss_train)

        with open(f"{folder_path}/loss_validation.npy", 'wb') as f:
            np.save(f, loss_validation)

        with open(f"{folder_path}/loss_train_cls.npy", 'wb') as f:
            np.save(f, loss_train_cls)

        with open(f"{folder_path}/loss_train_bb.npy", 'wb') as f:
            np.save(f, loss_train_bb)

        # create learning curves plot and save it
        fig, _ = self.show_learning_curves()
        fig.savefig(f"{folder_path}/loss_curves.png")

    def show_learning_curves(self):

        fig, ax = plt.subplots()
        ax.plot(range(len(self.train_loss)), self.train_loss)
        ax.plot(range(len(self.valid_loss)), self.valid_loss)
        ax.plot(range(len(self.train_loss_cls)), self.train_loss_cls)
        ax.plot(range(len(self.train_loss_bb)), self.train_loss_bb)
        plt.legend(["Training loss", "Validation loss", "Training loss classification", "Training loss bounding box"])
        plt.title("Training and validation loss")
        ax.grid("minor")
        plt.xlabel("epoch")
        plt.ylabel("loss")

        return fig, ax


def get_new_result_folder_path(path: str) -> str:
    """ Returns the path to store the model training results.

    The folders, in which the results of different models are saved, are numbered in ascending order (model_01, model_02, ...).
    All these model-folders are stored in one parent folder.
    This function returns the path to the folder with the next higher model-folder-number.

    Args:
        path: Path to the parent folder, in which all model_x folders are stored.

    Returns:
        folder_path: Path of a folder with the next higher model-folder-number.
    """

    folders = glob.glob(path + "/*/")

    if len(folders) == 0:
        folder_name = "model_01"

    else:
        numbers = []
        for folder in folders:
            i = 2
            while True:
                try:
                    number = int(folder[-i:-1])
                    i += 1
                except Exception:
                    break

            numbers.append(number)

        next_number = max(numbers) + 1
        if next_number < 10:
            next_number_str = f"0{str(next_number)}"
        else:
            next_number_str = str(next_number)

        folder_name = f"model_{next_number_str}"

    folder_path = f"{path}/{folder_name}"

    return folder_path


def set_seeds(seed: int) -> None:
    torch_geometric.seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
