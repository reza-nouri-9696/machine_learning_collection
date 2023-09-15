# machine_learning_collection
Overview
This GitHub repository contains code for a machine learning model training pipeline. The pipeline is designed to train and evaluate various machine learning models on a given dataset with k-fold cross-validation. It also allows for easy switching between different models using a model dispatcher and stores trained models for future use.

Repository Structure
The repository is organized as follows:

train.py: This script is the main entry point for training machine learning models. It takes two command-line arguments, --fold (the fold number for cross-validation) and --model (the name of the machine learning model to train). It loads the dataset, splits it into training and validation sets, trains the selected model, evaluates its performance, and saves the trained model to a file.

model_dispatcher.py: This module defines a dictionary of machine learning models, each associated with a unique name. The dictionary is used in train.py to select and initialize the specified model for training.

config.py: This module contains configuration settings, including the path to the training dataset (TRAINING_FILE) and the directory where trained models will be saved (MODEL_OUTPUT).

input/train_folds.csv: This is the training dataset with an additional column for k-fold cross-validation.
