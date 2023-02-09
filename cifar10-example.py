
# Imports
import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
# from multiprocessing import Pool  # , Process
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
import pendulum
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from ast import literal_eval

### Global configurables:


# How many of the samples in the data set to actually use on this training run
number_of_samples_to_use = 200
INPUT_SHAPES = (224, 224, 3)

# Cerebros configurables:

activation = 'relu'
predecessor_level_connection_affinity_factor_first = 2.0
predecessor_level_connection_affinity_factor_main = 0.97
max_consecutive_lateral_connections = 5
p_lateral_connection = 0.97
num_lateral_connection_tries_per_unit = 2
learning_rate = 0.001
epochs = 10  # [1, 100]
batch_size = 20
maximum_levels = 4  # [3,7]
maximum_units_per_level = 7  # [2,10]
maximum_neurons_per_unit = 4  # [2,20]


# Build BERT base model
image_input_0 = tf.keras.layers.Input(shape=INPUT_SHAPES[0])
preprocessor = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",
                              output_shape=[1001])
classifier_output = preprocessor(image_input_0)
foundation_model = tf.keras.Model(image_input_0,
                                  classifier_output)

for layer in foundation_model.layers:
    layer.trainable = True

relevant_layers = foundation_model.layers[-2]
embedding_model = tf.keras.Model(inputs=foundation_model.layers[0].input,
                    outputs=foundation_model.layers[-2].output)

##### Fix the data ingestion and the Cerebros config ...
### Load the Data set ... repalce with image ingestion
# raw_text = pd.read_csv(data_file, dtype='object')
# raw_text = raw_text.iloc[:number_of_samples_to_use, :]
# One hot encode the label
# raw_text[prediction_target_column] =\
#   raw_text[prediction_target_column]\
#   .apply(lambda x: 1 if x == positive_class_label else 0)
labels = '' # raw_text.pop(prediction_target_column)
labels = '' # labels.values
data = '' # raw_text.values

labels_tensor = tf.constant(labels, dtype=tf.int8)
data_tensor = tf.constant(data, dtype=tf.string)

TIME = pendulum.now(tz='America/New_York').__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_test'
INPUT_SHAPES = [()]

# Cerebros parameters:

training_x = [data_tensor]
train_labels = [labels_tensor]

OUTPUT_SHAPES = [1]
meta_trial_number = str(int(np.random.random() * 10 ** 12))

cerebros_automl = SimpleCerebrosRandomSearch(
    unit_type=DenseUnit,
    input_shapes=INPUT_SHAPES,
    output_shapes=OUTPUT_SHAPES,
    training_data=training_x,
    labels=train_labels,
    validation_split=0.35,
    direction='maximize',
    metric_to_rank_by="val_binary_accuracy",
    minimum_levels=2,
    maximum_levels=maximum_levels,
    minimum_units_per_level=1,
    maximum_units_per_level=maximum_units_per_level,
    minimum_neurons_per_unit=1,
    maximum_neurons_per_unit=maximum_neurons_per_unit,
    activation=activation,
    final_activation='sigmoid',
    number_of_architecture_moities_to_try=2,
    number_of_tries_per_architecture_moity=1,
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
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(),
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall()],
    epochs=epochs,
    project_name=f"{PROJECT_NAME}_meta_{meta_trial_number}",
    # use_multiprocessing_for_multiple_neural_networks=False,  # pull this param
    model_graphs='model_graphs',
    batch_size=batch_size,
    meta_trial_number=meta_trial_number,
    base_models=[embedding_model],
    train_data_dtype=tf.string)
val_binary_accuracy = cerebros_automl.run_random_search()
print(val_binary_accuracy)
