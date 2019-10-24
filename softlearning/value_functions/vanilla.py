import tensorflow as tf
from tensorflow.python.keras.engine import training_utils

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.utils.keras import PicklableModel


def create_feedforward_Q_function(input_shapes,
                                  *args,
                                  preprocessors=None,
                                  observation_keys=None,
                                  name='feedforward_Q',
                                  **kwargs):
    if 'meta_time' in input_shapes['observations'].keys():
        input_shapes['observations'].pop('meta_time')
        preprocessors['observations'].pop('meta_time')
    if 'desired_goal' in input_shapes['observations'].keys():
        input_shapes['observations'].pop('desired_goal')
        preprocessors['observations'].pop('desired_goal')
    input_shapes['observations']['env_latents'] = tf.TensorShape(2)
    preprocessors['observations']['env_latents'] = None
    inputs_flat = create_inputs(input_shapes)
    preprocessors_flat = (
        flatten_input_structure(preprocessors)
        if preprocessors is not None
        else tuple(None for _ in inputs_flat))

    assert len(inputs_flat) == len(preprocessors_flat), (
        inputs_flat, preprocessors_flat)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_
        in zip(preprocessors_flat, inputs_flat)
    ]

    Q_function = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs)

    Q_function = PicklableModel(inputs_flat, Q_function(preprocessed_inputs))
    Q_function.observation_keys = observation_keys

    return Q_function
