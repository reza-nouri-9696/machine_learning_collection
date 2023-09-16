you must add your dataset in this folder.
I use MNIST dataset.
notic that you have to add kfold column on your dataset. you can use this code to add column on your dataset:

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("your path/train.csv")
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    df.to_csv("your path/train_folds.csv")
