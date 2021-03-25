import tensorflow as tf
conv_dims=[5,5]
conv_height_index = tf.keras.backend.arange(0, stop=conv_dims[0])
conv_width_index = tf.keras.backend.arange(0, stop=conv_dims[1])
conv_height_index = tf.keras.backend.tile(conv_height_index, [conv_dims[1]])

# TODO: Repeat_elements and tf.split doesn't support dynamic splits.
# conv_width_index = tf.keras.backend.repeat_elements(conv_width_index, conv_dims[1], axis=0)
conv_width_index = tf.keras.backend.tile(
    tf.keras.backend.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
conv_width_index = tf.keras.backend.flatten(tf.keras.backend.transpose(conv_width_index))
# 361*2
conv_index = tf.keras.backend.transpose(tf.keras.backend.stack([conv_height_index, conv_width_index]))
print(conv_index)
conv_index = tf.keras.backend.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
print(conv_index)
