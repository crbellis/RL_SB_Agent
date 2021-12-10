import tensorflow as tf

def agent(state_shape, action_shape):
	""" The agent maps X-states to Y-actions
	e.g. The neural network output is [.1, .7, .1, .3]
	The highest value 0.7 is the Q-Value.
	The index of the highest action (0.7) is action #1.
	"""
	# intializing model with weights where samples are drawn from truncated normal distribution with mean of zero and std of sqrt(2/input)
	# see https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
	init = tf.keras.initializers.VarianceScaling(scale=2, mode='fan_in', distribution='truncated_normal')
	# creating densely connect neural net with input shape the size of the current board
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(128, input_shape=state_shape, activation='relu', kernel_initializer=init))
	model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer=init))
	model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer=init))
	model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer=init))
	model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer=init))
	# output shape of the model is the size of the action space, in the case of sokoban 4 elements are output
	# kernel initializers are random uniform distribution, starting between -0.03 and 0.03
	model.add(tf.keras.layers.Dense(action_shape, activation='linear', kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03)))
	# compiling model with MSE and Adam optimizer
	model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
	return model
