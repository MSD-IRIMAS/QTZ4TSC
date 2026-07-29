import tensorflow as tf

def inception_module(input_tensor,
                     stride=1,
                     activation='linear',
                     nb_filters=32,
                     use_residual=True,
                     use_bottleneck=True,
                     bottleneck_size=32,
                     kernel_size=40):
    if use_bottleneck and int(input_tensor.shape[-1]) > bottleneck_size:
        input_inception = tf.keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                          padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                         strides=stride, padding='same', activation=activation, use_bias=False)(
        input_inception))

    max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                     padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = tf.keras.layers.Concatenate(axis=2)(conv_list)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    return x


def shortcut_layer(input_tensor, output_tensor):
    shortcut_y = tf.keras.layers.Conv1D(filters=int(output_tensor.shape[-1]), kernel_size=1, padding='same', use_bias=False)(input_tensor)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)    
    
    x = tf.keras.layers.Add()([shortcut_y, output_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


def get_model(n_classes=32,
              input_shape=(None, 1),
              nb_filters=32,
              use_residual=True,
              use_bottleneck=True,
              depth=6,
              kernel_size=40):
    input_layer = tf.keras.layers.Input(input_shape)
    
    x = input_layer
    input_res = input_layer    
    
    for d in range(depth):    
        x = inception_module(x,
                             nb_filters=nb_filters,
                             use_residual=use_residual,
                             use_bottleneck=use_bottleneck,
                             kernel_size=kernel_size)
        if use_residual and d % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    output_layer = tf.keras.layers.Dense(units=n_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model
