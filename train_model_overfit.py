from yolo_model import DarkNet
import tensorflow as tf
import numpy as np
from utils import load_one_dataset
from yolo_loss import loss_function
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# tf.keras.backend.set_learning_phase(False)
# model = DarkNet()
#
# model(tf.keras.layers.Input(shape=(608, 608, 3)))
#
# print(model.summary())

# 加载模型
model = tf.saved_model.load('saved_model/yolo_model_file')


# 载入训练集合
# (1, 608, 608, 3) 和 (3, 5)
dog_image_data, dog_label = load_one_dataset('dog', '.jpg')
# (1, 608, 608, 3) 和 (1, 5)
car2_image_data, car2_label = load_one_dataset('car2', '.png')

# (2, 608, 608, 3)
images_data = np.concatenate([dog_image_data, car2_image_data], axis=0)

# 2个样本label  (2, ?, 5)
labels = [dog_label, car2_label]

# (Batch,19,19,425)
predictions = model(images_data)


# (Batch, 19, 19, 425) 和 (Batch, ?, 5)
loss_function(predictions, labels)
# 227.13795

