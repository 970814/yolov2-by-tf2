from yolo_model import DarkNet
from utils import parse_yolo_v2_model_weights, preprocess_image, generateXYOffset, readAnchorBoxShape, \
    draw_boxes, read_class_name, generate_colors
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

image_name = 'dog.jpg'
# 图片放缩成608,608,像素值归一化
image, image_data, image_shape = preprocess_image(img_path='images/' + image_name, model_image_size=(608, 608))

# 每个单元格的5个锚框的形状，维度为 (5,2)
anchor_box_shape = readAnchorBoxShape('./model_data/yolo_anchors.txt')
# 80个类别的名称
class_name = read_class_name('./model_data/coco_classes.txt')
# 进行向前传播算法
pred = model(image_data)
# N,19,19,5,85
predictions = tf.reshape(pred, [-1, 19, 19, 5, 85])
# 转换成更有意义的值
# 相对单元格的位置
# N,19,19,5,2
box_xy = tf.sigmoid(predictions[:, :, :, :, 0:2])
# 相对anchor-box宽高的系数
# N,19,19,5,2
box_wh = tf.exp(predictions[:, :, :, :, 2:4])
# 该位置包含对象的把握
# N,19,19,5,1
box_conf = tf.sigmoid(predictions[:, :, :, :, 4:5])
# 该位置包含的对象关于类别的概率分布
# N,19,19,5,80
box_class_prob = tf.nn.softmax(predictions[:, :, :, :, 5:85])
# 再转换
# 19,19,2      生成每个单元格相对整张表格的偏移
xy_offset = generateXYOffset()
# 1,19,19,1,2  同一个单元格下的所有anchor-box共用偏移
xy_offset2 = np.expand_dims(xy_offset, axis=[0, -2])
# 加上偏移，便得到相对整张表格的位置
box_xy = box_xy + xy_offset2
# 位置单位是每个单元格的长度，除整张表格的长度，得出一个比例0～1，相对整张表格的位置，或者说相对整张图的位置
# N,19,19,5,2
box_xy = box_xy / [19, 19]
# 每个单元格的5个锚框的形状
fixed_anchor_box_shape = np.reshape(anchor_box_shape, [1, 1, 1, 5, 2])
# box_wh是anchor—box宽高的系数，乘积得到真实的宽高，单位为单元格的长度
box_wh = box_wh * fixed_anchor_box_shape
# 得出一个比例0～1，宽高分别是相对整张表格的宽高
box_wh = box_wh / [19, 19]
# 转换成边角坐标
box_wh_half = box_wh / 2
xy_min = box_xy - box_wh_half
xy_max = box_xy + box_wh_half
# N,19,19,5,4
boxes = tf.keras.backend.concatenate([
    xy_min[:, :, :, :, 1:2],  # y_min
    xy_min[:, :, :, :, 0:1],  # x_min
    xy_max[:, :, :, :, 1:2],  # y_max
    xy_max[:, :, :, :, 0:1],  # x_max
], axis=-1)
# 计算出每个类别的得分，   实际上是转换成81类别的概率分布，其中80个是预定义类别，另外一个是不包含对象的概率
# N,19,19,5,1 *  N,19,19,5,80 -> N,19,19,5,80
box_scores = box_conf * box_class_prob
# 得分最高的类别当作该盒子的预测类别
# N,19,19,5   包含最高得分的类别
box_class = tf.argmax(box_scores, axis=-1)
# N,19,19,5   包含某个类别的最高得分
box_scores = tf.reduce_max(box_scores, axis=-1)

# 首先把得分低于0.6的框滤去
# N,19,19,5
obj_high_prob_mask = box_scores >= 0.6
# K,
box_high_scores = tf.boolean_mask(box_scores, obj_high_prob_mask)
# K,
box_high_scores_class = tf.boolean_mask(box_class, obj_high_prob_mask)
# N,19,19,5,4     N,19,19,5    ->  K,4
high_scores_boxes = tf.boolean_mask(boxes, obj_high_prob_mask)

# 因为存在多个框同时检测同一个对象的可能，
# 使用非最大值印制，当多个框同时检测同一个类别的同一个对象时，选择得分最高的框。
# 也就是说需要对不同的类别应用一次非最大值印制算法
box_index = tf.image.non_max_suppression(
    high_scores_boxes, box_high_scores, max_output_size=10, iou_threshold=0.1)
# print(box_index)
# 获取非最大值印制后的结果
# N,
res_score = tf.gather(box_high_scores, box_index)
# N,
res_class = tf.gather(box_high_scores_class, box_index)
# N,4
res_boxes = tf.gather(high_scores_boxes, box_index)

# image_shape 为原图大小高宽格式，
# 还原成相对于原图的位置
res_boxes = res_boxes * np.tile(list(image_shape), 2)
# 把框画在图片上
draw_boxes(image, res_score, res_boxes, res_class, class_name, generate_colors(class_name))
# plt.imshow(image)
# plt.show()
image.save('images/out/' + image_name, quality=100)


