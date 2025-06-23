import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data():
    dir_BakedPotato = Path(r'archive\Fast Food Classification V2\Train/Baked Potato')
    dir_Burger = Path(r'archive\Fast Food Classification V2\Train/Burger')
    dir_Fries = Path(r'archive\Fast Food Classification V2\Train/Fries')

    filepaths_BakedPotato = list(dir_BakedPotato.glob('**/*.JPEG'))
    filepaths_Burger = list(dir_Burger.glob('**/*.JPEG'))
    filepaths_Fries = list(dir_Fries.glob('**/*.JPEG'))

    filepaths = sum([filepaths_BakedPotato, filepaths_Burger, filepaths_Fries], [])
    list_BakedPotato = ['BakedPotato'] * len(filepaths_BakedPotato)
    list_Burger = ['Burger'] * len(filepaths_Burger)
    list_Fries = ['Fries'] * len(filepaths_Fries)
    labels = sum([list_BakedPotato, list_Burger, list_Fries], [])

    filepaths_S = pd.Series(filepaths, name='FilePaths')
    label_S = pd.Series(labels, name='labels')
    data = pd.merge(filepaths_S, label_S, left_index=True, right_index=True)

    data['FilePaths'] = data['FilePaths'].astype(str)
    X_train, X_test = train_test_split(data, test_size=0.15, stratify=data['labels'])
    X_train, X_val = train_test_split(X_train, test_size=0.2, stratify=X_train['labels'])

    return X_train, X_val, X_test
