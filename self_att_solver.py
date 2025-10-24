"This module provides the `SelfAttSolver` class, which encapsulates the training and evaluation logic for the baseline Transformer Encoder model with standard self-attention.

This solver is analogous to the `Solver` for the Anomaly Transformer and is used to train and test the baseline model for comparison purposes. It handles the data loading, model building, training loop, and evaluation, using the reconstruction error as the anomaly score."
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_factory.data_loader import SMDSegLoader, MSLSegLoader, SMAPSegLoader, PSMSegLoader
from self_attention.TransformerEncoder import TransformerEncoder
from utils import utils


def adjust_learning_rate(optimizer, epoch, lr_):
    """
    Adjusts the learning rate of the optimizer during training.

    The learning rate is decayed by a factor of 0.5 at specific epochs.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): The current epoch number.
        lr_ (float): The initial learning rate.
    """
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class SelfAttSolver(object):
    """
    A solver for training and evaluating the baseline Transformer Encoder model.

    This class handles the entire pipeline, including data loading, model construction, training, and testing. The anomaly detection is based on the reconstruction error of the model.
    """
    DEFAULTS = {}

    def __init__(self, config):
        """
        Initializes the SelfAttSolver.

        Args:
            config (dict): A dictionary containing the configuration parameters for the model, data, and training.
        """

        self.num_epochs = 0
        self.optimizer = None
        self.model = None
        self.lr = None
        self.d_model = None
        self.n_heads = None
        self.e_layers = None
        self.output_c = None
        self.input_c = None
        self.dataset = ''
        self.win_size = 0
        self.batch_size = None
        self.data_path = None
        self.model_save_path = ''
        self.criterion = None

        self.__dict__.update(SelfAttSolver.DEFAULTS, **config)

        self.model_checkpoint_path = os.path.join(self.model_save_path, str(self.dataset) + '_checkpoint.pth')

        # Initialize the data loaders for training, validation, and testing.
        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        """
        Builds the Transformer Encoder model and the optimizer.
        """
        self.model = TransformerEncoder(enc_in=self.input_c, c_out=self.output_c, d_model=self.d_model, n_heads=self.n_heads, e_layers=self.e_layers)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self):
        """
        Trains the Transformer Encoder model.

        The training process is based on minimizing the reconstruction error (MSE loss) between the input and the model's output.
        """
        print("====================== TRANSFORMER TRAIN MODE ======================")

        # Clean previous model checkpoint.
        if os.path.isfile(self.model_checkpoint_path):
            os.remove(self.model_checkpoint_path)
            print(f'Removed previous checkpoint at {self.model_checkpoint_path}')

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, _) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                # Get the reconstructed output from the model.
                output = self.model(input)
                # The loss is the reconstruction error (MSE).
                rec_loss = self.criterion(output, input)
                loss1_list.append(rec_loss.item())
                # Print training progress.
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                rec_loss.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} ".format(epoch + 1, train_steps, train_loss))
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        # Save the model checkpoint after training.
        if not os.path.isfile(self.model_checkpoint_path):
            torch.save(self.model.state_dict(), self.model_checkpoint_path)
            print(f"Saved model checkpoint at {self.model_checkpoint_path}")

        print("====================== END TRAINING ======================")

    def test(self):
        """
        Tests the trained Transformer Encoder model for anomaly detection.

        The anomaly score is the reconstruction error. The threshold is determined using the percentile of the combined energy of the training and test sets.

        Returns:
            dict: A dictionary containing the evaluation metrics (accuracy, precision, recall, F-score).
        """
        # Load the pre-trained model.
        self.model.load_state_dict(
                torch.load(
                        os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'), weights_only=True, ))
        self.model.eval()

        print("====================== TRANSFORMER TEST MODE ======================")

        test_criterion = nn.MSELoss(reduce=False)

        # (1) Calculate the reconstruction error on the training set.
        attens_energy = []
        for i, (input_data, _) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output = self.model(input)
            loss = torch.mean(test_criterion(input, output), dim=-1)
            loss = loss.detach().cpu().numpy()
            attens_energy.append(loss)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) Determine the anomaly threshold.
        attens_energy = []
        for i, (input_data, _) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output = self.model(input)
            loss = torch.mean(test_criterion(input, output), dim=-1)
            loss = loss.detach().cpu().numpy()
            attens_energy.append(loss)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        print("Threshold :", thresh)

        # (3) Evaluate the model on the test set.
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output = self.model(input)
            loss = torch.mean(test_criterion(input, output), dim=-1)
            loss = loss.detach().cpu().numpy()
            attens_energy.append(loss)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # Detection adjustment: a post-processing step to improve the detection of anomalous segments.
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        # Calculate and print the evaluation metrics.
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
                "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                        accuracy, precision,
                        recall, f_score))

        # Plot the sampled reconstruction error with the threshold.
        utils.generate_sampled_plot(test_energy, thresh, title='Reconstruction Error (Sampled)', sampling_rate=50)

        return {
            'accuracy':  accuracy,
            'precision': precision,
            'recall':    recall,
            'f_score':   f_score
        }


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    """
    Factory function to create a DataLoader for a specific time series dataset.

    Args:
        data_path (str): The path to the dataset directory.
        batch_size (int): The number of samples per batch.
        win_size (int): The size of the sliding window.
        step (int): The step size for the sliding window.
        mode (str): The mode of operation ('train', 'val', or 'test').
        dataset (str): The name of the dataset to load.

    Returns:
        DataLoader: A PyTorch DataLoader instance for the specified dataset.
    """
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader