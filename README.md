# machine_learning_collection
# Professor Mohammadali Javadzadeh javadzadeh@ihu.ac.ir
## Overview

This GitHub repository contains a Python project for automated machine learning model training. The project includes a set of scripts and configuration files to train and evaluate various machine learning models on a given dataset with k-fold cross-validation.

## Files and Directory Structure

The repository is organized as follows:

### `train.py`

This script is the main entry point for training machine learning models. It takes two command-line arguments: `--fold` (the fold number for k-fold cross-validation) and `--model` (the name of the machine learning model to train). It reads the training data from the CSV file specified in `config.py`, splits it into training and validation sets based on the fold, trains the selected model, and saves the trained model to the specified output directory.

### `model_dispatcher.py`

This module defines a dictionary of machine learning models that can be used in the `train.py` script. Each model is associated with a unique name, and the corresponding scikit-learn classifier is instantiated based on the selected name.

### `config.py`

This configuration file defines the paths to the training data CSV file and the directory where trained models will be saved.

### `input/`

This directory is not included in the repository but should contain the training data CSV file (`train_folds.csv`) used by the `train.py` script. You need to place your dataset in this directory or adjust the `TRAINING_FILE` path in `config.py` accordingly.

### `models/`

This directory is where the trained models will be saved by the `train.py` script.


To train a machine learning model using this project, follow these steps:

1. Place your training dataset in the `input/` directory or update the `TRAINING_FILE` path in `config.py` to point to your dataset.

2. Run the `train.py` script with the desired fold and model name. For example, to train a decision tree model using fold 0, run:

   ```shell
   python train.py --fold 0 --model decision_tree_gini
The script will train the selected model, print the accuracy on the validation set, and save the trained model to the models/ directory.

Repeat the above step with different fold numbers and model names to train and evaluate various models.

## Dependencies

This project relies on the following Python libraries:

- `pandas`
- `scikit-learn` (sklearn)
- `joblib`

Make sure to install these libraries using pip before running the project.

# Available Models

The following machine learning models are available for training:

- Decision Tree (Gini Impurity and Entropy criteria)
- Random Forest(rf)
- Multi-Layer Perceptron (Neural Network)(mlp)
- Support Vector Machine (Linear, Polynomial, RBF, and Sigmoid kernels)(svc(kernel='rbf'))
- Gaussian Naive Bayes
- k-Nearest Neighbors(knn)

# Results

After running the training script for different models and folds, you can evaluate model performance using metrics such as accuracy. The trained models will be saved in the specified `MODEL_OUTPUT` directory.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

This project was created as a part of a machine learning experiment and is intended for educational purposes.

Please feel free to fork and modify this repository for your own use or contribute to its development. If you have any questions or suggestions, feel free to open an issue or reach out to the repository owner.

Happy Machine Learning!
