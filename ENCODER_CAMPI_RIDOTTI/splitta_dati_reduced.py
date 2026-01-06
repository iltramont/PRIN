import pandas as pd
from pathlib import Path
#from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

from constants import AnnotationsReduced

import random


from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns



############
# Parameters
############
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 2026
DATA_FILE_NAME = "tumoreprimitivo_reduced_clean.csv"

TRAIN_SPLIT_FILE_NAME = 'train_split_reduced'
TEST_SPLIT_FILE_NAME = 'test_split_reduced'
VALIDATION_SPLIT_FILE_NAME = 'validation_split_reduced'
SAVE_DATA = True

REPORT__COLUMN_NAME = 'report_text'
ANNOTATOR_COLUMN_NAME = 'profile'
STRATIFY_COLUMNS = (
    'morfologia',
    'infiltrazione_sfinteri',
    'infiltrazione_tessuto_adiposo',
    'depositi_tumorali',
    'emvi_esteso',
    'stadio_T',
    'stadio_N',
    'stadio_N1c',
    'metastasi',
    'coinvolgimento_riflessione_peritoneale',
    'coinvolgimento_fascia_mesorettale',
    'infiltrazione_organi_extra'
)

#############
# Preliminari
#############
random.seed(RANDOM_STATE)
base_dir = Path(__file__).parent.parent
plt.style.use('ggplot')


##############
# Get raw data
##############
data_path = base_dir / "data" / DATA_FILE_NAME
data = pd.read_csv(data_path)
print(f'{data.shape = }')

# Keep only report and target columns
target_columns = list(AnnotationsReduced.model_fields.keys())


###########
# Splitting
###########
X = data[[REPORT__COLUMN_NAME, ANNOTATOR_COLUMN_NAME] + target_columns].copy(deep=True) 
print(f'Selezionate solo colonne di interesse\n{X.shape = }\n')
# Create dummies to stratify (train - test)
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X[list(STRATIFY_COLUMNS)])
Y_dummy = encoder.transform(X[list(STRATIFY_COLUMNS)])
print(f'Create dummies per stratificazione\n{Y_dummy.shape = }')
# Train test split with stratification
index_train, Y_train, index_test, Y_test = iterative_train_test_split(X.index.to_numpy().reshape(-1, 1),
                                                                      Y_dummy,
                                                                      test_size=TEST_SIZE)
index_train = index_train.reshape(1, -1)[0]
index_test = index_test.reshape(1, -1)[0]
X_train = X.iloc[index_train].copy(deep=True)
X_test = X.iloc[index_test].copy(deep=True)

# Create dummies to stratify (train - validation)
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[list(STRATIFY_COLUMNS)])
Y_dummy = encoder.transform(X_train[list(STRATIFY_COLUMNS)])
print(f'Create dummies per stratificazione\n{Y_dummy.shape = }')
# Train validation split with stratification
index_train, Y_train, index_validation, Y_validation = iterative_train_test_split(X_train.index.to_numpy().reshape(-1, 1),
                                                                                Y_dummy,
                                                                                test_size=(VALIDATION_SIZE / (1 - TEST_SIZE)))
index_train = index_train.reshape(1, -1)[0]
index_validation = index_validation.reshape(1, -1)[0]
X_validation = X_train.loc[index_validation].copy(deep=True)
X_train = X_train.loc[index_train].copy(deep=True)

print(f'Selezionate solo colonne di interesse per lo split di validazione\n{X_validation.shape = }\n')

X_train['split'] = 'train'
X_test['split'] = 'test'
X_validation['split'] = 'validation'

print(f'\nSplitatto il dataset\n{X_train.shape = }\n{X_test.shape = }\n{X_validation.shape = }')


##########################
# Visualize stratification
##########################
n_columns = 3
n_rows, r = divmod(len(STRATIFY_COLUMNS), n_columns)
if r != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, n_columns)
fig.suptitle("Stratification", fontsize='xx-large')

df = pd.concat([X_train, X_test, X_validation]).fillna('NaN')
for i in range(len(STRATIFY_COLUMNS)):
    sns.countplot(data=df, x=STRATIFY_COLUMNS[i], hue='split', ax=axes[i//n_columns][i%n_columns])

plt.show()


###########
# Save data
###########
if SAVE_DATA:
    train_path = base_dir / "data" / (TRAIN_SPLIT_FILE_NAME + '.csv')
    X_train.to_csv(train_path, index=False)
    test_path = base_dir / "data" / (TEST_SPLIT_FILE_NAME + '.csv')
    X_test.to_csv(test_path, index=False)
    validation_path = base_dir / "data" / (VALIDATION_SPLIT_FILE_NAME + '.csv')
    X_validation.to_csv(validation_path, index=False)
