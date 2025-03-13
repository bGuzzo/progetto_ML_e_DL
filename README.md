### Anomaly Transformer: Experimental Project for the Machine and Deep Learning Exam

#### Introduction

This repository contains the code and experiments related to my experimental project on the Anomaly Transformer, an attention-based architecture for anomaly detection in time series. The project was carried out for the Machine and Deep Learning exam of the A.Y. 2023/2024.

#### Get Started

*   Use Python 3.10/3.12 and run the script /requirements/install_pkgs.sh
*   Use a venv or other environment (e.g., Conda) to avoid installing global packages!

#### Repository Content

*   `model/`: PyTorch implementation of the Anomaly Transformer.
*   `self_attention/`: PyTorch implementation of a model with classic self-attention for Anomaly Detection
*   `solver.py`: Script to train and test the Anomaly Transformer model.
*   `self_att_solver.py`: Script to train and test the model with classic self-attention.
*   `grid_search.py`: (Main file) Implementation of the Grid Search for Anomaly Transformer.
*   `grid_search_self_att.py`: (Main file) Implementation of the Grid Search for the model with classic self-attention.
*   `results/`: Folder to save the results of the grid search execution
*   `relazione/`: Folder containing the project report (PDF) and presentation
*   `requirements/install_pkgs.sh`: Bash script to install the necessary packages (torch and others)
*   `requirements/requirements.txt`: Python requirements file, used by the `install_pkgs.sh` script

#### Experiments and Comparisons

The project included the following experiments and comparisons:

*   **Hyperparameter Sensitivity Analysis:** Study of the impact of hyperparameters on the model, particularly with reduced dimensions.
*   **Optimization Algorithms:** Comparison of the performance of different optimization algorithms (Adam, AdamW, SGD, Adadelta, RMSprop).
*   **RNN in combination with Anomaly Transformer:** Evaluation of the use of LSTM in combination with the Anomaly Transformer.
*   **Anomaly Attention vs. Self Attention:** Direct comparison between the Anomaly Attention mechanism and classic Self Attention.

#### Results

The results of the experiments are reported in the project report.