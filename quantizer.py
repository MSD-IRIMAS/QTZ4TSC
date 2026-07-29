import numpy as np
import tensorflow as tf


def apply(model, qtz):
    weights = model.get_weights()
    new_weights = [qtz(w) for w in weights]
    model.set_weights(new_weights)
    return model


def apply_to_conv(model, qtz):
    for layer in model.layers:
        weights = layer.get_weights()

        if isinstance(layer, tf.keras.layers.Conv1D):
            new_weights = [qtz(w) for w in weights]
        else:
            continue

        layer.set_weights(new_weights)
    return model


# only for test, set all weights to 0
def zero_quantize(x):
    x = np.zeros(x.shape, dtype=x.dtype)
    return x


def identity_quantize(x):
    x = np.copy(x)
    return x


def int_quantize(x, bits):
    max_int = 2**(bits - 1) - 1
    scale = np.max(np.abs(x)) / max_int
    q = np.round(x / scale)
    q = np.clip(q, -max_int - 1, max_int)
    return (q * scale).astype(np.float32)


def int_quantize_with_zero(x, bits):
    max_int = 2**(bits) - 1
    zero = np.min(x)
    dynamic = np.max(x) - np.min(x)
    scale = dynamic / max_int
    q = x - zero
    q = np.round(q / scale)
    q = np.clip(q, 0, max_int)
    return ((q * scale) + zero).astype(np.float32)


def int_quantize_with_zero_no_outliers(x, bits, outlier_thresh):
    max_int = 2**(bits) - 1
    zero = max(np.min(x),-outlier_thresh)
    range_max = min(np.max(x),outlier_thresh)
    dynamic = range_max - zero
    scale = dynamic / max_int
    quantize_values = np.copy(x) #Yeah it's bad
    for i in range(len(quantize_values)):
        if(np.abs(quantize_values[i]) <= outlier_thresh):
            q = quantize_values[i] - zero
            q = np.round(q / scale)
            q = np.clip(q, 0, max_int)
            quantize_values[i] = ((q * scale)+zero).astype(np.float32)
    return quantize_values
