import tensorflow as tf

def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.VarianceScaling(scale=2, mode='fan_in', distribution='truncated_normal')
	# tf.keras.initializers.HeUniform()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(action_shape, activation='linear'))
    # model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    model.compile(loss="huber", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model
