import numpy as np
import tensorflow as tf
# 2,2,2
t1 = tf.Variable(np.array([[[1, 2], [2, 3]], [[4, 4], [5, 3]]]))
# 2,2,2
t2 = tf.Variable(np.array([[[7, 4], [8, 4]], [[2, 10], [15, 11]]]))

# 4,2,2
d0 = tf.keras.layers.concatenate([t1, t2], axis=0)
# 2,4,2
d1 = tf.keras.layers.concatenate([t1, t2], axis=1)
# 2,2,4
d2 = tf.keras.layers.concatenate([t1, t2], axis=2)
d3 = tf.keras.layers.concatenate([t1, t2], axis=-1)

print(d0)
print(d1)
print(d2)
print(d3)

