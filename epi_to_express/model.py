import tensorflow as tf

def conv_profile_task_base(output_shape,window_size=20_000,pred_res=100, 
                           bottleneck=8, activation='relu',
                           output_activation='softplus'):
    """
    CODE ADAPTED FROM https://www.biorxiv.org/content/10.1101/2022.04.29.490059v1.full
    
    Task-specific convolutional model
    :param input_shape: tuple of input shape
    :param output_shape: tuple of output shape
    :param bottleneck: bottleneck size
    :param pred_resolution: resolution of histone mark data
    :return: model
    
    NOTE: can only handle one input resolution for now
    """
    #three pred res options
    pred_res_opts = [100,500,2000]
    if (type(pred_res)==list):
        print("Warning: model can only handle one prediction resolution, using first")
        pred_res = pred_res[0]
    assert pred_res in pred_res_opts, f"pred_res must be one of {pred_res_opts}"
    output_len, num_tasks = output_shape
    #output_len = output_shape[0]
    inputs = dict(
        x_p_pred_res= tf.keras.layers.Input(
            shape=(window_size//pred_res, 1),
            name=f'x_p_pred_res'
        )
    )
    nn = tf.keras.layers.Conv1D(filters=192, kernel_size=19, 
                                padding='same')(inputs['x_p_pred_res'])
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation(activation, name='filter_activation')(nn)
    nn = tf.keras.layers.MaxPool1D(pool_size=8)(nn)
    nn = tf.keras.layers.Dropout(0.1)(nn)

    nn = tf.keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = tf.keras.layers.Dropout(0.1)(nn)
    
    nn = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same')(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = tf.keras.layers.Dropout(0.2)(nn)

    nn = tf.keras.layers.Flatten()(nn)
    nn = tf.keras.layers.Dense(256)(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.Dropout(0.3)(nn)

    nn = tf.keras.layers.Dense(output_len * bottleneck)(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.Reshape([output_len, bottleneck])(nn)
    nn = tf.keras.layers.Dropout(0.1)(nn)

    nn = tf.keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.Dropout(0.2)(nn)

    nn2 = tf.keras.layers.Conv1D(filters=64, kernel_size=7, padding='same')(nn)
    nn2 = tf.keras.layers.BatchNormalization()(nn2)
    nn2 = tf.keras.layers.Activation('relu')(nn2)
    nn2 = tf.keras.layers.Dropout(0.1)(nn2)
    outputs = tf.keras.layers.Dense(1)(nn2)
    outputs = tf.keras.layers.Dense(1,activation=output_activation)(nn2)

    return tf.keras.Model(inputs=inputs, outputs=outputs,name="CNN_mod")

def residual_block(input_layer, kernel_size=3, activation='relu', num_layers=5, dropout=0.1):
    factor = range(1,num_layers)
    base_rate = 2
    """dilated residual convolutional block"""
    filters = input_layer.shape.as_list()[-1]
    nn = tf.keras.layers.Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             activation=None,
                             padding='same',
                             dilation_rate=1)(input_layer)
    nn = tf.keras.layers.BatchNormalization()(nn)
    for i in factor:
        nn = tf.keras.layers.Activation('relu')(nn)
        nn = tf.keras.layers.Dropout(dropout)(nn)
        nn = tf.keras.layers.Conv1D(filters=filters,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 dilation_rate=base_rate ** i)(nn)
        nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.add([input_layer, nn])
    return tf.keras.layers.Activation(activation)(nn)

def residual_profile_task_base(input_shape, output_shape, bottleneck=8, 
                               output_activation="softplus"):
    """
    CODE ADAPTED FROM https://www.biorxiv.org/content/10.1101/2022.04.29.490059v1.full
    
    residual model with task specific heads at base resolution
    :param input_shape: tuple of input shape
    :param output_shape: tuple of output shape
    :param bottleneck: bottleneck size
    :return: model
    """
    output_len, num_tasks = output_shape
    #output_len = output_shape[0]

    inputs = tf.keras.Input(shape=input_shape, name='assay')

    nn = tf.keras.layers.Conv1D(filters=192, kernel_size=19, padding='same')(inputs)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu', name='filter_activation')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=5)
    nn = tf.keras.layers.MaxPool1D(pool_size=8)(nn)
    nn = tf.keras.layers.Dropout(0.1)(nn)

    nn = tf.keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=5)
    nn = tf.keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = tf.keras.layers.Dropout(0.1)(nn)

    nn = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same')(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=3)
    nn = tf.keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = tf.keras.layers.Dropout(0.2)(nn)

    nn = tf.keras.layers.Flatten()(nn)
    nn = tf.keras.layers.Dense(512)(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.Dropout(0.3)(nn)

    nn = tf.keras.layers.Dense(output_len * bottleneck)(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.Reshape([output_len, bottleneck])(nn)
    nn = tf.keras.layers.Dropout(0.1)(nn)

    nn = tf.keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation('relu')(nn)
    nn = tf.keras.layers.Dropout(0.2)(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=6)

    nn2 = tf.keras.layers.Conv1D(filters=64, kernel_size=7, padding='same')(nn)
    nn2 = tf.keras.layers.BatchNormalization()(nn2)
    nn2 = tf.keras.layers.Activation('relu')(nn2)
    nn2 = tf.keras.layers.Dropout(0.1)(nn2)
    outputs = tf.keras.layers.Dense(1,activation=output_activation)(nn2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs,name="resid_mod")
    return model
