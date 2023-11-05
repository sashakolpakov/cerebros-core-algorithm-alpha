# -*- coding: utf-8 -*-
"""phishing-email-detection-gpt2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10KKTHjBkdfKBpT9OLIj2eZs533BuCS6h

## GPT2 + Cerebros for Phishing email detection

Initialization
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/MyDrive/Colab\ Notebooks/cerebros-core-algorithm-alpha

!pip install -r requirements.txt

!pip install -q --upgrade keras-nlp

import tensorflow as tf
import tensorflow_text
from keras_nlp.models import GPT2Tokenizer, GPT2Preprocessor, GPT2Backbone
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten
import pandas as pd
import numpy as np
from cerebros.simplecerebrosrandomsearch.simple_cerebros_random_search\
    import SimpleCerebrosRandomSearch
import pendulum
from cerebros.units.units import DenseUnit
from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component\
    import zero_7_exp_decay, zero_95_exp_decay, simple_sigmoid
from ast import literal_eval

#
# Load the email data
#
df = pd.read_csv("Phishing_Email.csv")
#
# Get the rows where 'Email Text' is a string, remove everything else
#
df = df[df['Email Text'].apply(lambda x: isinstance(x, str))]
#
# Reset the index
#
df.reset_index(drop=True, inplace=True)

#
# Binary label for email type: positive type is "phishing"
#
label_mapping = {"Safe Email": 0, "Phishing Email": 1}
df["Binary Label"] = df["Email Type"].map(label_mapping)
#
# Data and labels ready
#
X = df["Email Text"].to_numpy()
y = df["Binary Label"].to_numpy()
#
# Shuffle the data
#
X, y = shuffle(X, y)

# Train / test split : we give 65% of the data for *testing*
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.65, shuffle=False)

#
# Tensors for training data and labels
#
training_x   = [tf.constant(X_train)]
train_labels = [tf.constant(y_train)]
#
# Input and output shapes
#
INPUT_SHAPES  = [()]
OUTPUT_SHAPES = [1]

"""### A custom GPT2 encoder layer for text embedding"""

class GPT2Layer(tf.keras.layers.Layer):

    def __init__(self, max_seq_length, **kwargs):
        #
        super(GPT2Layer, self).__init__(**kwargs)
        #
        # Load the GPT2 tokenizer, preprocessor and model
        self.tokenizer = GPT2Tokenizer.from_preset("gpt2_base_en")
        self.preprocessor = GPT2Preprocessor(self.tokenizer,
                                             sequence_length=max_seq_length)
        self.encoder   = GPT2Backbone.from_preset("gpt2_base_en")
        #
        # Set whether the GPT2 model's layers are trainable
        #self.encoder.trainable = False
        for layer in self.encoder.layers:
            layer.trainable = False
        #
        self.encoder.layers[-2].trainable = True
        #
        # Set the maximum sequence length for tokenization
        self.max_seq_length = max_seq_length

    def call(self, inputs):
        #
        # Output the GPT2 embedding
        prep = self.preprocessor([inputs])
        embedding  = self.encoder(prep)
        avg_pool = tf.reduce_mean(embedding, axis=1)
        #
        return avg_pool

    def get_config(self):
        #
        config = super(GPT2Layer, self).get_config()
        config.update({'max_seq_length': self.max_seq_length})
        #
        return config

    @classmethod
    def from_config(cls, config):
        #
        return cls(max_seq_length=config['max_seq_length'])

# GPT2 configurables
max_seq_length = 96

# Base model
input_layer = Input(shape=(), dtype=tf.string)
gpt2_layer = GPT2Layer(max_seq_length)(input_layer)
#output = Flatten()(gpt2_layer)
base_model = Model(inputs=input_layer, outputs=gpt2_layer)
base_model.summary()

"""### Cerebros search for the best model"""

#
# Cerebros configurables
#
activation = 'swish'
predecessor_level_connection_affinity_factor_first = 2.0
predecessor_level_connection_affinity_factor_main = 0.97
max_consecutive_lateral_connections = 5
p_lateral_connection = 0.97
num_lateral_connection_tries_per_unit = 2
learning_rate = 0.001
epochs = 6  # [1, 100]
batch_size = 32
maximum_levels = 5  # [3,7]
maximum_units_per_level = 7  # [2,10]
maximum_neurons_per_unit = 6  # [2,20]

#
# Logging
#
TIME = pendulum.now(tz='America/New_York').__str__()[:16]\
    .replace('T', '_')\
    .replace(':', '_')\
    .replace('-', '_')
PROJECT_NAME = f'{TIME}_cerebros_auto_ml_phishing_email_test'

meta_trial_number = 42 # irrelevant unless in distributed training

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
    model_graphs='model_graphs',
    batch_size=batch_size,
    meta_trial_number=meta_trial_number,
    base_models=[base_model],
    train_data_dtype=tf.string)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# result = cerebros_automl.run_random_search()

print(f'Best accuracy achieved is {result}')
print(f'binary accuracy')

"""### Testing the best model found"""

#
# Load the best model (taking into account that it has a custom layer)
#
best_model_found =\
tf.keras.models.load_model(cerebros_automl.best_model_path,\
custom_objects={'GPT2Layer': GPT2Layer(max_seq_length)})

best_model_found.evaluate(X_test, y_test)

"""### Training the best model on a larger dataset, and testing again"""

# Train / test split : we give 65% of the data for training,
# now that we have found the best model
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.35, shuffle=False)

optimizer = Adam(learning_rate=0.0005)
#loss=tf.keras.losses.BinaryCrossentropy()
loss = tf.keras.losses.CategoricalHinge()
metrics=[tf.keras.metrics.BinaryAccuracy(),
         tf.keras.metrics.Precision(),
         tf.keras.metrics.Recall()]
best_model_found.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

best_model_found.fit(X_train, y_train, validation_split=0.35, epochs=3)

best_model_found.evaluate(X_test, y_test)