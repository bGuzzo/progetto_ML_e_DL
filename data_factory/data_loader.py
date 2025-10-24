"""
This module provides data loaders for various time series datasets used in the Anomaly Transformer project.

It includes specific loader classes for the PSM, MSL, SMAP, and SMD datasets, which handle the loading, preprocessing, and segmentation of the time series data. The preprocessing steps include standardization using StandardScaler and handling of missing values. The data is then segmented into windows of a specified size, which can be used for training, validation, and testing of the Anomaly Transformer model.

The main function `get_loader_segment` acts as a factory to get the appropriate data loader for a given dataset.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


class PSMSegLoader(object):
    """
    Data loader for the PSM (Pooled Server Metrics) dataset.

    This class handles the loading, preprocessing, and segmentation of the PSM dataset.
    The data is loaded from CSV files, standardized using StandardScaler, and segmented into windows.
    """
    def __init__(self, data_path, win_size, step, mode="train"):
        """
        Initializes the PSMSegLoader.

        Args:
            data_path (str): The path to the directory containing the dataset files.
            win_size (int): The size of the sliding window used for segmentation.
            step (int): The step size for the sliding window.
            mode (str): The mode of operation, one of "train", "val", or "test".
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Returns the total number of segments in the dataset.

        The number of segments is calculated based on the mode (train, val, or test) and the sliding window parameters.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Retrieves a single segment and its corresponding label from the dataset.

        Args:
            index (int): The index of the segment to retrieve.

        Returns:
            tuple: A tuple containing the data segment and its label, both as float32 numpy arrays.
        """
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                    self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                    self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    """
    Data loader for the MSL (Mars Science Laboratory) dataset.

    This class handles the loading, preprocessing, and segmentation of the MSL dataset.
    The data is loaded from .npy files, standardized using StandardScaler, and segmented into windows.
    """
    def __init__(self, data_path, win_size, step, mode="train"):
        """
        Initializes the MSLSegLoader.

        Args:
            data_path (str): The path to the directory containing the dataset files.
            win_size (int): The size of the sliding window used for segmentation.
            step (int): The step size for the sliding window.
            mode (str): The mode of operation, one of "train", "val", or "test".
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Returns the total number of segments in the dataset.

        The number of segments is calculated based on the mode (train, val, or test) and the sliding window parameters.
        """

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Retrieves a single segment and its corresponding label from the dataset.

        Args:
            index (int): The index of the segment to retrieve.

        Returns:
            tuple: A tuple containing the data segment and its label, both as float32 numpy arrays.
        """
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                    self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                    self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    """
    Data loader for the SMAP (Soil Moisture Active Passive) dataset.

    This class handles the loading, preprocessing, and segmentation of the SMAP dataset.
    The data is loaded from .npy files, standardized using StandardScaler, and segmented into windows.
    """
    def __init__(self, data_path, win_size, step, mode="train"):
        """
        Initializes the SMAPSegLoader.

        Args:
            data_path (str): The path to the directory containing the dataset files.
            win_size (int): The size of the sliding window used for segmentation.
            step (int): The step size for the sliding window.
            mode (str): The mode of operation, one of "train", "val", or "test".
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Returns the total number of segments in the dataset.

        The number of segments is calculated based on the mode (train, val, or test) and the sliding window parameters.
        """

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Retrieves a single segment and its corresponding label from the dataset.

        Args:
            index (int): The index of the segment to retrieve.

        Returns:
            tuple: A tuple containing the data segment and its label, both as float32 numpy arrays.
        """
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                    self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                    self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    """
    Data loader for the SMD (Server Machine Dataset) dataset.

    This class handles the loading, preprocessing, and segmentation of the SMD dataset.
    The data is loaded from .npy files, standardized using StandardScaler, and segmented into windows.
    """
    def __init__(self, data_path, win_size, step, mode="train"):
        """
        Initializes the SMDSegLoader.

        Args:
            data_path (str): The path to the directory containing the dataset files.
            win_size (int): The size of the sliding window used for segmentation.
            step (int): The step size for the sliding window.
            mode (str): The mode of operation, one of "train", "val", or "test".
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):
        """
        Returns the total number of segments in the dataset.

        The number of segments is calculated based on the mode (train, val, or test) and the sliding window parameters.
        """

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Retrieves a single segment and its corresponding label from the dataset.

        Args:
            index (int): The index of the segment to retrieve.

        Returns:
            tuple: A tuple containing the data segment and its label, both as float32 numpy arrays.
        """
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                    self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                    self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    """
    Factory function to create a DataLoader for a specific time series dataset.

    This function selects the appropriate data loader class based on the `dataset` argument,
    and creates a DataLoader instance with the specified parameters.

    Args:
        data_path (str): The path to the dataset directory.
        batch_size (int): The number of samples per batch.
        win_size (int): The size of the sliding window. Defaults to 100.
        step (int): The step size for the sliding window. Defaults to 100.
        mode (str): The mode of operation ('train', 'val', or 'test'). Defaults to 'train'.
        dataset (str): The name of the dataset to load. One of 'SMD', 'MSL', 'SMAP', 'PSM'. Defaults to 'KDD'.

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