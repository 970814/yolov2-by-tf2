from yolo_model import DarkNet
from utils import parse_yolo_v2_model_weights, preprocess_image, readAnchorBoxShape, \
    draw_boxes, read_class_name, generate_colors, convert_filter_and_non_max_suppression
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# tf.keras.backend.set_learning_phase(False)
model = DarkNet()
# 编译网络，生成weights参数结构
model(tf.keras.layers.Input(shape=(608, 608, 3)))

print(model.summary())
yolo_weights = parse_yolo_v2_model_weights(model.weight_shapes, 'yolov2_weights/yolov2.weights')
model.set_weights(yolo_weights)

image_name = 'person.jpg'
# 图片放缩成608,608,像素值归一化
image, image_data, image_shape = preprocess_image(img_path='images/' + image_name, model_image_size=(608, 608))

# 80个类别的名称
class_name = read_class_name('./model_data/coco_classes.txt')
# 进行向前传播算法
pred = model(image_data)

res_class, res_score, res_boxes = convert_filter_and_non_max_suppression(pred)

# image_shape 为原图大小高宽格式，
# 还原成相对于原图的位置
res_boxes = res_boxes * np.tile(list(image_shape), 2)
# 把框画在图片上
draw_boxes(image, res_score, res_boxes, res_class, class_name, generate_colors(class_name))
# plt.imshow(image)
# plt.show()
image.save('images/out/' + image_name, quality=100)


