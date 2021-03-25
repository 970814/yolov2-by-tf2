from yolo_model import DarkNet
from utils import parse_yolo_v2_model_weights, preprocess_image
import tensorflow as tf

# tf.keras.backend.set_learning_phase(False)
model = DarkNet()
# 编译网络，生成weights参数
# model.build((None, 608, 608, 3))
model(tf.keras.layers.Input(shape=(608, 608, 3)))

print(model.summary())
# yolo_weights = parse_yolo_v2_model_weights(model.weight_shapes, 'yolov2.weights')
# model.set_weights(yolo_weights)

# 图片放缩成608,608,像素值归一化
# image, image_data, image_shape = preprocess_image(img_path='images/car2.png', model_image_size=(608, 608))
# predictions = model(image_data, )

# print(predictions.shape)
