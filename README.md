# machine_learning_collection
Overview
This GitHub repository contains code for a machine learning model training pipeline. The pipeline is designed to train and evaluate various machine learning models on a given dataset with k-fold cross-validation. It also allows for easy switching between different models using a model dispatcher and stores trained models for future use.

Repository Structure
The repository is organized as follows:

train.py: This script is the main entry point for training machine learning models. It takes two command-line arguments, --fold (the fold number for cross-validation) and --model (the name of the machine learning model to train). It loads the dataset, splits it into training and validation sets, trains the selected model, evaluates its performance, and saves the trained model to a file.

model_dispatcher.py: This module defines a dictionary of machine learning models, each associated with a unique name. The dictionary is used in train.py to select and initialize the specified model for training.

config.py: This module contains configuration settings, including the path to the training dataset (TRAINING_FILE) and the directory where trained models will be saved (MODEL_OUTPUT).

input/train_folds.csv: This is the training dataset with an additional column for k-fold cross-validation. Notice that this dataset was modified and you must add a column called k-fold.you can use this code to modify MNIST dataset:
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("/path your dataset/train.py")
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    df.to_csv("path your dataset/train_folds.csv")


Getting Started
To use this repository, follow these steps:

Clone the repository to your local machine:

bash
git clone https://github.com/your-username/machine-learning-pipeline.git
Install the required dependencies, which may include scikit-learn and joblib:

bash
pip install scikit-learn joblib
Customize the configuration in config.py according to your dataset and preferences.

Run the training script train.py with the desired fold and model. For example:

bash
python train.py --fold 0 --model decision_tree_gini
This command will train a decision tree classifier with the Gini impurity criterion on fold 0 of the dataset.

Available Models
The following machine learning models are available for training:

Decision Tree (Gini Impurity and Entropy criteria)
Random Forest
Multi-Layer Perceptron (Neural Network)
Support Vector Machine (Linear, Polynomial, RBF, and Sigmoid kernels)
Gaussian Naive Bayes
k-Nearest Neighbors
Results
After running the training script for different models and folds, you can evaluate model performance using metrics such as accuracy. The trained models will be saved in the specified MODEL_OUTPUT directory.

License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed.

Acknowledgments
The code structure and organization are inspired by best practices in machine learning model training pipelines.
The project uses scikit-learn for machine learning model implementation.
Contact
For any questions or feedback, please contact reza nouri at rezanouri9696@gmail.com.

Happy machine learning!
