from functools import partial, update_wrapper
from keras.metrics import top_k_categorical_accuracy


# This is needed to keep the __name__ attribute of the partial func (needed in Keras)
def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def top_k(k):
    return wrapped_partial(top_k_categorical_accuracy, k=k)
