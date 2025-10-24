# Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

This repository provides a comprehensive implementation and an in-depth experimental analysis of the Anomaly Transformer, a state-of-the-art deep learning architecture for unsupervised anomaly detection in time series data. This project is based on the seminal paper ["Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy" by Xu et al. (2022)](https://arxiv.org/abs/2110.02642).

Beyond a faithful implementation of the original model, this repository extends the initial research by conducting a series of rigorous experiments to validate and further understand the behavior of the Anomaly Transformer. The key contributions of this work include:

- **Hyperparameter Sensitivity Analysis:** A detailed investigation into the model's performance under different hyperparameter configurations, particularly with reduced model dimensionality and fewer training epochs, to assess its robustness and efficiency.
- **Comparative Analysis of Optimization Algorithms:** An empirical study comparing the performance and training time of various optimization algorithms (e.g., Adam, SGD, RMSprop) for training the Anomaly Transformer.
- **Architectural Exploration:** Experiments with architectural modifications, such as the integration of LSTM layers, to explore potential performance enhancements.
- **Head-to-Head with Standard Self-Attention:** A direct and fair comparison between the Anomaly Transformer's novel Anomaly-Attention mechanism and a baseline Transformer Encoder using standard self-attention, quantifying the performance gains of the proposed approach.

This work was developed as a research project for the Machine and Deep Learning exam at the University of Calabria, and it aims to provide a thorough and well-documented resource for researchers and practitioners interested in advanced time series anomaly detection.

## Table of Contents
- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
  - [Association Discrepancy](#association-discrepancy)
  - [Minimax Learning Strategy](#minimax-learning-strategy)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Anomaly Transformer](#training-the-anomaly-transformer)
  - [Grid Search](#grid-search)
- [Experiments and Key Findings](#experiments-and-key-findings)
  - [Hyperparameter Sensitivity Analysis](#hyperparameter-sensitivity-analysis)
  - [Optimization Algorithm Comparison](#optimization-algorithm-comparison)
  - [Architectural Modifications: Integrating RNNs](#architectural-modifications-integrating-rnns)
  - [Anomaly Attention vs. Standard Self-Attention](#anomaly-attention-vs-standard-self-attention)
- [Conclusion](#conclusion)
- [The Anomaly Transformer Paper](#the-anomaly-transformer-paper)

## The Anomaly Transformer Paper

This project is an implementation and extension of the research paper **["Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy"](https://arxiv.org/abs/2110.02642)** by Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long, presented at ICLR 2022. This section provides a detailed summary of the original paper, which serves as the foundation for this work.

### Abstract

The paper introduces the Anomaly Transformer, a novel framework for unsupervised anomaly detection in time series. The authors observe that traditional methods, which rely on pointwise representations or pairwise associations, are often insufficient for capturing the intricate dynamics of time series data. They leverage the power of Transformers to model both pointwise representations and pairwise associations in a unified manner.

The core idea is that anomalies, being rare, struggle to form strong associations with the entire series and instead tend to associate more with their immediate temporal neighbors. This "adjacent-concentration bias" is exploited to create a distinguishable criterion called **Association Discrepancy**. The Anomaly Transformer, featuring a novel **Anomaly-Attention** mechanism, is proposed to compute this discrepancy. A minimax optimization strategy is employed to amplify the distinguishability between normal and anomalous points. The paper demonstrates state-of-the-art performance on six unsupervised time series anomaly detection benchmarks.

### Introduction

The paper begins by highlighting the challenges of unsupervised time series anomaly detection, particularly the difficulty in learning informative representations from complex temporal dynamics and deriving a criterion that can effectively separate rare anomalies from normal data. It critiques existing methods, including classic statistical models and deep learning approaches based on RNNs, for their limitations in capturing long-range dependencies and providing a comprehensive temporal context.

The authors propose to adapt Transformers for this task, capitalizing on their ability to model global representations and long-range relationships. They introduce the concept of **series-association**, which is the attention distribution of a time point over the entire series, and **prior-association**, which is an inductive bias that assumes anomalies primarily associate with adjacent points. The discrepancy between these two associations forms the basis of their anomaly detection criterion.

### Methodology

The paper details the architecture and training strategy of the Anomaly Transformer:

*   **Anomaly-Attention Mechanism:** A two-branch self-attention mechanism is proposed.
    *   The **prior-association branch** uses a learnable Gaussian kernel to model the adjacent-concentration bias.
    *   The **series-association branch** learns associations from the raw data using standard self-attention.
*   **Association Discrepancy:** This is formalized as the symmetrized KL divergence between the prior- and series-associations, averaged over all layers of the model.
*   **Minimax Association Learning:** A minimax strategy is used to optimize the model.
    *   **Minimize Phase:** The prior-association is trained to approximate the series-association, allowing it to adapt to the data's temporal patterns.
    *   **Maximize Phase:** The series-association is optimized to maximize the discrepancy from the prior-association, forcing it to focus on non-adjacent patterns and making anomalies more distinguishable.
*   **Association-based Anomaly Criterion:** The final anomaly score is a combination of the normalized Association Discrepancy and the reconstruction error, allowing both temporal representation and association discrepancy to contribute to the detection.

### Experiments

The paper evaluates the Anomaly Transformer on six benchmarks, including SMD, PSM, MSL, SMAP, SWaT, and a new NeurIPS-TS benchmark. The model is compared against 18 baselines, including reconstruction-based, density-estimation, clustering-based, and autoregression-based methods.

The results show that the Anomaly Transformer consistently achieves state-of-the-art performance across all datasets. The ablation studies confirm the effectiveness of each component of the model, including the association-based criterion, the learnable prior-association, and the minimax strategy.

### Conclusion

The paper concludes that the Anomaly Transformer, with its novel Anomaly-Attention mechanism and minimax learning strategy, provides a powerful and effective solution for unsupervised time series anomaly detection. The proposed Association Discrepancy criterion is shown to be highly effective in distinguishing anomalies from normal data.


## Introduction

Time series anomaly detection is a critical task in various domains, such as system monitoring, finance, and industrial maintenance. Traditional methods, including statistical approaches and Recurrent Neural Networks (RNNs), often struggle to capture long-range dependencies and effectively distinguish rare anomalies from normal fluctuations.

The Anomaly Transformer addresses these limitations by introducing a novel attention-based neural network architecture. It is designed for unsupervised anomaly detection and leverages a key insight: anomalies, due to their rarity, have difficulty establishing significant associations with the entire time series, concentrating their relationships primarily on adjacent time points. This "adjacent-concentration bias" provides an intrinsic criterion for discriminating between normal and anomalous data points.

## Key Concepts

### Association Discrepancy

The core of the Anomaly Transformer is the **Association Discrepancy**. Instead of relying solely on the reconstruction error, the model computes two types of associations:

1.  **Prior-Association:** A prior belief about the attention distribution, based on a Gaussian kernel that models the "adjacent-concentration bias". This gives more weight to adjacent time points.
2.  **Series-Association:** Learned from the data using a standard scaled dot-product attention mechanism, capturing the complex dependencies within the time series.

The **Association Discrepancy**, calculated as the Kullback-Leibler (KL) divergence between the Prior-Association and the Series-Association, serves as a crucial component of the anomaly score. For normal time points, the two associations are expected to be similar, resulting in a small discrepancy. For anomalies, the discrepancy is expected to be large.

### Minimax Learning Strategy

To amplify the Association Discrepancy, the Anomaly Transformer employs a minimax learning strategy during training:

-   **Minimization Phase:** The model is trained to minimize the reconstruction error while also minimizing the KL divergence between the Prior-Association and the (detached) Series-Association. This guides the Prior-Association to adapt to the temporal patterns in the data.
-   **Maximization Phase:** The model is trained to minimize the reconstruction error while maximizing the KL divergence between the Series-Association and the (detached) Prior-Association. This pushes the Series-Association to focus on long-range dependencies, making it harder to reconstruct anomalies and thus increasing the discrepancy.

The final anomaly score is a combination of the reconstruction error and the Association Discrepancy, providing a more robust and accurate measure for anomaly detection.

## Repository Structure

```
.
├── data_factory
│   ├── data_loader.py
│   └── __init__.py
├── dataset
│   └── MSL
│       ├── MSL_test_label.npy
│       ├── MSL_test.npy
│       └── MSL_train.npy
├── grid_search.py
├── grid_search_self_att.py
├── LICENSE
├── main.py
├── model
│   ├── AnomalyTransformer.py
│   ├── attn.py
│   ├── embed.py
│   ├── __init__.py
│   ├── kernel.py
│   ├── loss_func.py
│   └── optimizer.py
├── README.md
├── relazione
│   ├── Presentazione_ML_&_DL.pdf
│   ├── Relazione_ML_&_DL.pdf
│   └── Relazione_ML_&_DL.txt
├── requirements
│   ├── install_pkgs.sh
│   └── requirements.txt
├── scripts
│   ├── MSL_cust.sh
│   └── MSL.sh
├── self_attention
│   ├── self_att_encoder.py
│   ├── self_attention.py
│   └── TransformerEncoder.py
├── self_att_solver.py
└── solver.py
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/progetto_ML_e_DL.git
    cd progetto_ML_e_DL
    ```

2.  **Install the required packages:**
    You can use the provided shell script to install the necessary Python packages.
    ```bash
    bash requirements/install_pkgs.sh
    ```
    Alternatively, you can install the packages manually from `requirements.txt`:
    ```bash
    pip install -r requirements/requirements.txt
    ```
### Frameworks
This project is built using the following main frameworks:
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [NumPy](https://numpy.org/doc/)
- [scikit-learn](https://scikit-learn.org/stable/documentation.html)


## Usage

### Training the Anomaly Transformer

The `main.py` script is the main entry point for training and testing the Anomaly Transformer. You can configure the model and training parameters using command-line arguments.

**Example:**
```bash
python main.py --dataset MSL --data_path ./dataset/MSL --input_c 55 --output_c 55 --win_size 100 --d_model 64 --e_layers 3 --n_heads 8 --num_epochs 2 --batch_size 128 --lr 1e-4 --k 3 --mode train
```

The `scripts` directory also contains shell scripts for running experiments with pre-defined configurations.

### Grid Search

The `grid_search.py` script allows you to perform a grid search to find the optimal hyperparameters for the Anomaly Transformer. The hyperparameter space can be configured in the `MSL_params` dictionary within the script.

**To run the grid search:**
```bash
python grid_search.py
```
The results will be saved to a log file in the `results` directory.

A similar script, `grid_search_self_att.py`, is available for the baseline Transformer Encoder model with standard self-attention.

## Experiments and Key Findings

A series of experiments were conducted to analyze the performance and robustness of the Anomaly Transformer. The key findings are summarized below. For a more detailed analysis, please refer to the research report in `relazione/Relazione_ML_&_DL.pdf`.

### Hyperparameter Sensitivity Analysis

The model's sensitivity to the `d_model` (dimensionality) and the number of training epochs was evaluated.

| `d_model` | `epoch` | Accuracy | Precision | Recall | F-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 512 | 3 | 0.9845 | 0.9182 | 0.9364 | 0.9272 |
| 64 | 2 | 0.9866 | 0.9206 | 0.9551 | 0.9375 |

The results show that even with a significant reduction in the model's dimensionality (from 512 to 64) and fewer training epochs, the Anomaly Transformer maintains excellent predictive performance. This highlights the model's robustness and efficiency.

### Optimization Algorithm Comparison

Different optimization algorithms were compared to evaluate their impact on performance and training time.

| Optimizer | Accuracy | Precision | Recall | F-Score | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Adam** | 0.9864 | 0.9195 | 0.9540 | 0.9364 | **199.22** |
| AdamW | 0.9862 | 0.9189 | 0.9533 | 0.9358 | 199.38 |
| SGD | 0.9897 | 0.9215 | 0.9858 | 0.9526 | 199.65 |
| Adadelta | 0.9876 | 0.9199 | 0.9663 | 0.9425 | 200.28 |
| RMSprop | 0.9858 | 0.9180 | 0.9499 | 0.9337 | 200.73 |

As expected, the **Adam** optimizer was the most performant in terms of training time, confirming its suitability for training this type of model.

### Architectural Modifications: Integrating RNNs

An experiment was conducted to investigate if integrating an LSTM network in place of the feed-forward network in the encoder layers could improve performance.

| RNN Layers | Accuracy | Precision | Recall | F-Score | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 (Original) | 0.9874 | 0.9198 | 0.9650 | 0.9419 | 199.91 |
| **1** | 0.9893 | 0.9203 | 0.9834 | **0.9508** | 202.93 |
| 2 | 0.9875 | 0.9176 | 0.9688 | 0.9425 | 208.68 |
| 4 | 0.9880 | 0.9213 | 0.9690 | 0.9445 | 216.32 |
| 8 | 0.9849 | 0.9197 | 0.9390 | 0.9292 | 234.41 |
| 16 | 0.9872 | 0.9189 | 0.9632 | 0.9405 | 271.91 |

While a single LSTM layer showed a slight improvement in the F-score, further analysis revealed that this was likely due to the inherent randomness in the implementation. The overall results confirm the robustness of the original Anomaly Transformer architecture, which does not require the additional complexity of RNNs.

### Anomaly Attention vs. Standard Self-Attention

This experiment compares the Anomaly Transformer with a baseline Transformer Encoder that uses standard self-attention.

| Model | `d_model` | `epoch` | Accuracy | Precision | Recall | F-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Self-attention | 128 | 3 | 0.8516 | 0.4049 | 0.8698 | 0.5526 |
| **Anomaly Attention** | 128 | 3 | 0.9883 | 0.9202 | 0.9735 | **0.9461** |
| Self-attention | 512 | 5 | 0.7999 | 0.3190 | 0.7917 | 0.4547 |
| **Anomaly Attention** | 512 | 5 | 0.9841 | 0.9183 | 0.9318 | **0.9250** |

The results demonstrate the clear superiority of the Anomaly Attention mechanism. The Anomaly Transformer achieves a **~43% improvement in F-score** compared to the standard self-attention model. The low precision of the self-attention model indicates a high number of false positives, which is effectively addressed by the Anomaly Transformer.

The following plots from the research report illustrate the difference in the anomaly scores. The Anomaly Score from the Anomaly Transformer provides a much clearer and more stable signal for identifying anomalies compared to the noisy reconstruction error of the standard self-attention model.

*(The plots are described in the `relazione/Relazione_ML_&_DL.pdf` document, showing a cleaner separation between normal and anomalous points for the Anomaly Transformer.)*

## Conclusion

The Anomaly Transformer represents a significant advancement in unsupervised anomaly detection for time series. By leveraging the novel concepts of Association Discrepancy and a minimax learning strategy, the model can effectively capture complex temporal dependencies and accurately distinguish rare anomalies from normal data fluctuations. The experimental results demonstrate the superiority of this architecture over previous methods, opening up new possibilities for the application of deep learning in real-world anomaly detection scenarios.
