import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_factory.data_loader import SMDSegLoader, MSLSegLoader, SMAPSegLoader, PSMSegLoader
from self_attention.TransformerEncoder import TransformerEncoder


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class SelfAttSolver(object):
    DEFAULTS = {}

    def __init__(self, config):

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
        self.model = TransformerEncoder(enc_in=self.input_c, c_out=self.output_c, d_model=self.d_model, n_heads=self.n_heads, e_layers=self.e_layers)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self):
        print("====================== TRANSFORMER TRAIN MODE ======================")

        # Clean previous model checkpoint
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
                # Call model and get reconstruction
                output = self.model(input)
                # Compute loss -> use reconstruction error only to train the model
                rec_loss = self.criterion(output, input)
                loss1_list.append(rec_loss.item())
                # Print metrics
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

        # Train End -> Save model checkpoint if not exists
        # Model not saved yet
        if not os.path.isfile(self.model_checkpoint_path):
            torch.save(self.model.state_dict(), self.model_checkpoint_path)
            print(f"Saved model checkpoint at {self.model_checkpoint_path}")

        print("====================== END TRAINING ======================")

    # Same as Anomaly Transformer
    def test(self):
        # Load pre-trained model
        self.model.load_state_dict(
                torch.load(
                        os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'), weights_only=True, ))
        self.model.eval()

        print("====================== TRANSFORMER TEST MODE ======================")

        test_criterion = nn.MSELoss(reduce=False)

        # (1) statistic on the train set
        # Compute the Anomaly Score
        attens_energy = []
        for i, (input_data, _) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output = self.model(input)
            loss = torch.mean(test_criterion(input, output), dim=-1)
            loss = loss.detach().cpu().numpy()
            attens_energy.append(loss)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # Final Anomaly-Score for training set
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, _) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output = self.model(input)
            loss = torch.mean(test_criterion(input, output), dim=-1)
            loss = loss.detach().cpu().numpy()
            attens_energy.append(loss)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        # Merged Reconstruction Error for training and model set
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        # Assume (100 - self.anomaly_ratio)% of anomalies in the dataset
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        # Model evaluation as Anomaly Transformer but using the reconstruction error
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

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
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

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
                "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                        accuracy, precision,
                        recall, f_score))

        # return accuracy, precision, recall, f_score
        return {
            'accuracy':  accuracy,
            'precision': precision,
            'recall':    recall,
            'f_score':   f_score
        }


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
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
