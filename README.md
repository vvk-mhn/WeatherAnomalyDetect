# Federated Extreme Weather Detection

Codebase for the CS-E4740 Federated Learning '26 project.

## Overview
This project formulates an unsupervised anomaly detection task over FMI Open Data as a Graph Total Variation Minimization (GTVMin) problem. The focus is to identify extreme weather events (e.g. storms/frost) using autoencoder reconstruction error, where node collaboration (federated learning) assists stations with scarce local anomaly representation.

### Data Assumption
Raw FMI weather observations should be downloaded via your `load_data.py` into a `./fmi_data/` repository as `.csv` files. The project relies strictly on four derived variables (`Air temperature`, `Relative humidity`, `Wind speed`, `Precipitation amount`).

## Structure

- **`config/`**: Contains `default_config.yaml` specifying alpha, architectures, & graph params.
- **`src/`**: 
  - `data_interface.py`: Handles CSV parsing.
  - `preprocessing.py`: Forms rolling windows and constructs ground truth anomaly labels for test phase.
  - `graph.py`: Employs correlation-of-temperature as an adjacency metric (System design choice).
  - `models/autoencoder.py`: PyTorch Autoencoders (Small & Large variants).
  - `fl_training.py`: Executable loop applying the fixed-point operator synchronous projection step `param.grad.add_(2 * alpha * coupling)`.
  - `local_baseline.py`: Local PyTorch training decoupled.
  - `evaluation.py`: Extract predictive errors and evaluates area metrics (ROC, PR).

## Execution

1. Make sure PyTorch and dependencies are installed (`pip install torch pandas numpy scikit-learn matplotlib pyyaml`).
2. Populate data using `load_data.py`.
3. Run `python run_experiments.py` to train Systems A, B, and C computationally.
4. Run `python analyze_results.py` to compare their relative AUC on predicting storms/extreme environments.

## Course Mapping Guarantees
- Unsupervised MSE acts as $L_i$.
- Coupling is represented as an explicit graph momentum step utilizing $\alpha$ hyperparameter.
- Validates 3 Systems: Local (System A), FedSmall (System B), FedLarge (System C)
