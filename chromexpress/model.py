import tensorflow as tf

def covnet(num_tasks=1,window_size=6_000,pred_res=100,
           filtsize = 8, poolsize = 4,
           output_activation='linear'):
    """
    Convolutional neural Network Model
    Similar to covnet model proposed by DeepChrome:
    https://github.com/QData/DeepChrome
    """
    nstates = [175,625,125]
    #three pred res options
    pred_res_opts = [100,500,2000]
    if (type(pred_res)==list):
        print("Warning: model can only handle one prediction resolution, using first")
        pred_res = pred_res[0]
    assert pred_res in pred_res_opts, f"pred_res must be one of {pred_res_opts}"
    inputs = dict(
        x_p_pred_res= tf.keras.layers.Input(
            shape=(window_size//pred_res, 1),
            name='x_p_pred_res'
        )
    )
    # 3 convolutional layers, 2 Dense layers, output layer
    # stage 1: Conv layers
    x = tf.keras.layers.Conv1D(filters=nstates[0], kernel_size=filtsize,
                               padding='same')(inputs['x_p_pred_res'])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool1D(poolsize)(x)
    #small amount dropout after conv useful-http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf
    x = tf.keras.layers.Dropout(0.1)(x)
    # ----
    x = tf.keras.layers.Conv1D(filters=nstates[0]*2, kernel_size=filtsize//2,
                               padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool1D(poolsize//2)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    # ----
    x = tf.keras.layers.Conv1D(filters=nstates[0]*4, kernel_size=filtsize//2,
                               padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool1D(poolsize//2)(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    # stage 2: Dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(nstates[1])(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(nstates[2])(x)
    x = tf.keras.layers.Activation('relu')(x)

    #output layer
    outputs = tf.keras.layers.Dense(num_tasks,activation=output_activation)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs,name="covnet")