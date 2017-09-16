from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras import regularizers

import logging


def get_model(num_classes, input_shape=(224, 224, 3),
              k_lambda=None, a_lambda=None):
    """
    Get the pre-trained ResNet50 model.
    :param num_classes: Number of output classes
    :param input_shape: Shape of the input (default: (224, 224, 3))
    :return: ResNet50 model
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    x = base_model.output
    x = Flatten()(x)
    if k_lambda is None:
        kernel_regularizer = None
    else:
        logging.info("Regularizing last layer kernel with L2 lambda={}.".format(k_lambda))
        kernel_regularizer = regularizers.l2(k_lambda)
    if a_lambda is None:
        activity_regularizer = None
    else:
        logging.info("Regularizing last activation layer with L1 lambda={}.".format(a_lambda))
        activity_regularizer = regularizers.l1(a_lambda)

    predictions = Dense(num_classes, activation='softmax', name='predictions',
                        kernel_regularizer=kernel_regularizer,
                        activity_regularizer=activity_regularizer)(x)
    model = Model(inputs=[base_model.input], outputs=[predictions])
    return model
