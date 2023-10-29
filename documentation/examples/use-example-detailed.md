

From a shell on a suitable Linux machine (tested on Ubuntu 22.04):

Clone the repo

`git clone https://github.com/david-thrower/cerebros-core-algorithm-alpha.git`

cd into it: `cd cerebros-core-algorithm-alpha`

install all required packages: `pip3 install -r requirements.txt`

cd into it: `test-cases/ames-housing-price-prediction`

Run the Ames housing data example:

`python3 ames_housing_pred.py`

You can also access a Jupyter notebook version of it as `ames_housing_pred.ipynb`

Let's look at the example: `ames_housing_pred.py`, which is in the main folder of this repo:

Import packages

```python3

import sys
sys.path.insert(0, '../..')

import numpy as np
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
import pendulum
import pandas as pd
import tensorflow as tf
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from ast import literal_eval
```

Set how much compute resources you want to spend (Cerebros will build and train a number of models that is the product of these 2 numbers)

Set up project and load data

```python3


# Set a project name:


TIME = pendulum.now().__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_test'


# Read in the data

raw_data = pd.read_csv('ames.csv')

# Rather than doing elaborate preprocessing, let's just drop all the columns
# that aren't numbers and impute 0 for anything missing

needed_cols = [col for col in raw_data.columns if raw_data[col].dtype != 'object']
data_numeric = raw_data[needed_cols].fillna(0).astype(float)
label = raw_data.pop('price')

# Convert to numpy
data_np = data_numeric.values

# Convert to a tensor
tensor_x = tf.constant(data_np)

# Define the training set and labels
training_x = [tensor_x]
train_labels = [label.values]

# Shape of the trining data [number of rows, number of columns]
INPUT_SHAPES = [training_x[i].shape[1] for i in np.arange(len(training_x))]

# Labels are a list of numbers, shape is the length of it
OUTPUT_SHAPES = [1]  
```

Cerebros hyperparameters
```python3

# Parameters for Cebros training (Approximately the optima 
# discovered in a Bayesian tuning study done on Katib
# for this data set)

# In distributed training set this to a random number, otherwise,
# you can just set it to 0. (it keeps file names unique when this runs multiple
# times with the same project, like we would in distributed training.)

meta_trial_number = 0  # In distributed training set this to a random number

# For the rest of these parameters, these are the tunable hyperparameters.
# We recommend searching a broad but realistic search space on these using a
# suitable tuner such as Katib on Kubeflow, Optuna, ray, etc.

activation = "gelu"
predecessor_level_connection_affinity_factor_first = 19.613
predecessor_level_connection_affinity_factor_main = 0.5518
max_consecutive_lateral_connections = 34
p_lateral_connection = 0.36014
num_lateral_connection_tries_per_unit = 11
learning_rate = 0.095
epochs = 145
batch_size = 634
maximum_levels = 5
maximum_units_per_level = 5
maximum_neurons_per_unit = 25

```

Instantiate an instance of Cerebros Neural Architecture Search (NAS)

```python3

cerebros =\
    SimpleCerebrosRandomSearch(
        unit_type=DenseUnit,
        input_shapes=INPUT_SHAPES,
        output_shapes=OUTPUT_SHAPES,
        training_data=training_x,
        labels=train_labels,
        validation_split=0.35,
        direction='minimize',
        metric_to_rank_by='val_root_mean_squared_error',
        minimum_levels=1,
        maximum_levels=maximum_levels,
        minimum_units_per_level=1,
        maximum_units_per_level=maximum_units_per_level,
        minimum_neurons_per_unit=1,
        maximum_neurons_per_unit=maximum_neurons_per_unit,
        activation=activation,
        final_activation=None,
        number_of_architecture_moities_to_try=7,
        number_of_tries_per_architecture_moity=1,
        number_of_generations=3,
        minimum_skip_connection_depth=1,
        maximum_skip_connection_depth=7,
    predecessor_level_connection_affinity_factor_first=predecessor_level_connection_affinity_factor_first,
        predecessor_level_connection_affinity_factor_first_rounding_rule='ceil',
            predecessor_level_connection_affinity_factor_main=predecessor_level_connection_affinity_factor_main,
        predecessor_level_connection_affinity_factor_main_rounding_rule='ceil',
        predecessor_level_connection_affinity_factor_decay_main=zero_7_exp_decay,
        seed=8675309,
        max_consecutive_lateral_connections=max_consecutive_lateral_connections,
        gate_after_n_lateral_connections=3,
        gate_activation_function=simple_sigmoid,
        p_lateral_connection=p_lateral_connection,
        p_lateral_connection_decay=zero_95_exp_decay,
        num_lateral_connection_tries_per_unit=num_lateral_connection_tries_per_unit,
        learning_rate=learning_rate,
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
        epochs=epochs,
        patience=7,
        project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
        model_graphs='model_graphs',
        batch_size=batch_size,
        meta_trial_number=meta_trial_number)

```

Run the Neural Architecture Search and get results back. Load the best model found by the search.

```python3
result = cerebros.run_random_search()
best_model_found = cerebros.get_best_model()
```

Output the number of trainable and non-trainable parameters of the model. Output the best RMSE.

```python3
trainable_params = np.sum([np.prod(w.get_shape()) for w in best_model_found.trainable_weights])
non_trainable_params = np.sum([np.prod(w.get_shape()) for w in best_model_found.non_trainable_weights])
total_params = trainable_params + non_trainable_params

print(f"Best model found: {total_params} total parameters ({trainable_params} trainable, {non_trainable_params} non-trainable)")

print(f"Best rmse is (val_root_mean_squared_error): {result}")
```

## Example output from this task:

- Ames housing data set, no pre-processing, no scaling.
- House sell price predictions, val_rmse $169.04592895507812.
- The mean sale price in the data was $180,796.06.
- Val set RMSE was 0.00935% the mean sale price. No, there's not an extra 0 in there, yes, you are reading it right.
- There was no pre-trained base model. The data in [ames.csv](ames.csv) is the only data any of the model's weights has ever seen.                                                      