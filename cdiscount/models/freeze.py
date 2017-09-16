
def set_trainable_layers(model, nb_layers):
    """
    Set the last nb_layers as trainable.
    Others layers are frozen.
    """
    for layer in model.layers:
        layer.trainable = False
    if nb_layers != 0:
        for layer in model.layers[-nb_layers:]:
            layer.trainable = True
