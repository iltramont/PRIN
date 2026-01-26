import pandas as pd
from pathlib import Path
#from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

import constants
import model_utils

import random

import ast

from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


############
# Parameters
############
SAVE_DATA = False


#############
# Preliminari
#############
random.seed(constants.SEED)
base_dir = Path(__file__).parent.parent
# Set plot style
plt.style.use('ggplot')
#sns.set_palette('hls')

# Colors
finomnia_palette = sns.color_palette(('#db038a',   # Pink
                                      '#66218a',   # Violet
                                      '#2659ab',   # Blue
                                      '#081c36',   # Dark blue
                                      '#45c9f5'))  # Light blue
sns.set_palette(finomnia_palette)


##############
# Get raw data
##############
data_path = base_dir / "data" / constants.CLEAN_DATA_FILE_NAME
data = pd.read_csv(data_path)
print(f'{data.shape = }')

# Keep only report and target columns
target_columns = list(constants.Annotations.model_fields.keys())


###########
# Splitting
###########
X = data[[constants.REPORT_COLUMN_NAME, constants.ANNOTATOR_COLUMN_NAME] + target_columns].copy(deep=True)
print(f'Selezionate solo colonne di interesse\n{X.shape = }\n')
# Create dummies to stratify (train - test)
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X[list(constants.STRATIFY_COLUMNS)])
Y_dummy = encoder.transform(X[list(constants.STRATIFY_COLUMNS)])
print(f'Create dummies per stratificazione\n{Y_dummy.shape = }')
# Train test split with stratification
index_train, Y_train, index_test, Y_test = iterative_train_test_split(X.index.to_numpy().reshape(-1, 1),
                                                                      Y_dummy,
                                                                      test_size=constants.TEST_SIZE)
index_train = index_train.reshape(1, -1)[0]
index_test = index_test.reshape(1, -1)[0]
X_train = X.iloc[index_train].copy(deep=True)
X_test = X.iloc[index_test].copy(deep=True)

# Create dummies to stratify (train - validation)
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[list(constants.STRATIFY_COLUMNS)])
Y_dummy = encoder.transform(X_train[list(constants.STRATIFY_COLUMNS)])
print(f'Create dummies per stratificazione\n{Y_dummy.shape = }')
# Train validation split with stratification
index_train, Y_train, index_validation, Y_validation = iterative_train_test_split(X_train.index.to_numpy().reshape(-1, 1),
                                                                                  Y_dummy,
                                                                                  test_size=(constants.VALIDATION_SIZE / (1 - constants.TEST_SIZE)))
index_train = index_train.reshape(1, -1)[0]
index_validation = index_validation.reshape(1, -1)[0]
X_validation = X_train.loc[index_validation].copy(deep=True)
X_train = X_train.loc[index_train].copy(deep=True)

print(f'Selezionate solo colonne di interesse per lo split di validazione\n{X_validation.shape = }\n')

X_train['split'] = 'train'
X_test['split'] = 'test'
X_validation['split'] = 'validation'

assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_train.index).isdisjoint(X_validation.index)
assert set(X_test.index).isdisjoint(X_validation.index)


print(f'\nSplitatto il dataset\n{X_train.shape = }\n{X_test.shape = }\n{X_validation.shape = }')


##########################
# Visualize stratification
##########################
plot_columns = model_utils.get_classification_fields(constants.Annotations) + model_utils.get_binary_classification_fields(constants.Annotations)

n_columns = 3
n_rows, r = divmod(len(plot_columns), n_columns)
if r != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, n_columns)
axes = axes.flatten()
fig.suptitle("Stratification", fontsize='xx-large')

df = pd.concat([X_train, X_test, X_validation]).fillna('NaN')
for i in range(len(plot_columns)):
    sns.countplot(data=df, x=plot_columns[i], hue='split', ax=axes[i], hue_order=['train', 'validation', 'test'])


# Rimuove eventuali assi vuoti
for j in range(len(plot_columns), len(axes)):
    fig.delaxes(axes[j])

plt.show()


############
# Multilabel
############

plot_columns = model_utils.get_multiple_choice_fields(constants.Annotations)

n_columns = 2
n_rows, r = divmod(len(plot_columns), n_columns)
if r != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, n_columns, figsize=(21, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(plot_columns):
    df_plot = pd.DataFrame(columns=[col, 'split'])
    for _, row in df.iterrows():
        l = ast.literal_eval(row[col])
        if len(l) == 0:
            continue
        else:
            split = row['split']
            for s in l:
                df_plot.loc[len(df_plot)] = [s, split]
    sns.countplot(data=df_plot, ax=axes[i], x=col, hue='split', hue_order=['train', 'validation', 'test'])
    # Add values on top of bars

# Rimuove eventuali assi vuoti
for j in range(len(plot_columns), len(axes)):
    fig.delaxes(axes[j])

plt.show()


###########
# Save data
###########
if SAVE_DATA:
    train_path = base_dir / "data" / (constants.TRAIN_SPLIT_FILE_NAME + '.csv')
    X_train.to_csv(train_path, index=False)
    test_path = base_dir / "data" / (constants.TEST_SPLIT_FILE_NAME + '.csv')
    X_test.to_csv(test_path, index=False)
    validation_path = base_dir / "data" / (constants.VALIDATION_SPLIT_FILE_NAME + '.csv')
    X_validation.to_csv(validation_path, index=False)
