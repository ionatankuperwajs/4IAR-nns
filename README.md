# 4IAR-nns

**Implementation of the deep neural networks described in Kuperwajs, Schütt, and Ma (2023).**

## Approach

This repository implements networks as models of human play in 4-in-a-row. The networks are trained and tested on human decisions in large-scale data. Trained networks are available upon request, and the cognitive model code is avilabile at https://github.com/ionatankuperwajs/4IAR-improvements.

## File description

- `network.py`: architecture for the neural networks
- `preprocessing.py` and `custom_dataset.py`: data formatting
- `load_train.py` and `training.py`: training scripts
- `load_test.py` and `testing.py`: testing scripts
- `analysis.py` and `summary_stats.py`: analysis scripts for figures in the paper
- `model_preprocessing.py`, `model_comparison.py`, and `model_improvements.py`: data formatting and analysis for the cognitive model comparison
- `train_network.sh`, `train_network_array.sh`, and `test_network.sh`: bash scripts for training and testing on the computing cluster
