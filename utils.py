"""Miscellaneous utility functions."""

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import random
import colorsys


def parse_yolo_v2_model_weights(weight_shapes, weights_path):
    weights_file = open(weights_path, 'rb')
    # 丢弃前16个字节
    weights_header = np.ndarray(
        shape=(4,), dtype='int32', buffer=weights_file.read(16))
    print(weights_header)
    # yolo v2 模型weights保存的格式为 [bias / beta, [gamma, mean, variance], conv_weights]
    # tensorflow 模型weights格式为  [conv_weights , [gamma, beta, mean, variance] / bias]
    yolo_weights = []
    # 统计解析了多少参数
    count = 0
    for i in range(len(weight_shapes)):
        conv_weights_shape = weight_shapes[i]
        # DarkNet conv_weights are serialized Caffe-style:
        # (out_dim, in_dim, height, width)
        # We would like to set these to Tensorflow order:
        # (height, width, in_dim, out_dim)
        yolo_weights_shape = (conv_weights_shape[-1], conv_weights_shape[-2],
                              conv_weights_shape[0], conv_weights_shape[1])
        print('parse shape', conv_weights_shape)
        print('channels', weight_shapes[i][-1])
        channels = weight_shapes[i][-1]
        # 乘起来计算一共有多少个weight参数
        weights_size = np.product(conv_weights_shape)

        conv_bias = np.ndarray(
            shape=(channels,), dtype='float32', buffer=weights_file.read(channels * 4))
        args = [conv_bias]
        if i < len(weight_shapes) - 1:
            # 如果不是最后一层卷积，那么都采用了bn
            bn_args = np.ndarray(
                shape=(3, channels), dtype='float32', buffer=weights_file.read(3 * channels * 4))
            args = [
                bn_args[0],  # scale gamma
                conv_bias,  # shift gamma
                bn_args[1],  # running mean
                bn_args[2],  # running var
            ]

        yolo_weight = np.ndarray(
            shape=yolo_weights_shape, dtype='float32', buffer=weights_file.read(weights_size * 4))
        conv_weights = np.transpose(yolo_weight, [2, 3, 1, 0])

        count += weights_size
        yolo_weights.append(conv_weights)
        for j in range(len(args)):
            yolo_weights.append(args[j])
            count += len(args[j])
    remaining_args = len(weights_file.read()) // 4
    print('读取了 ', count, '/', count + remaining_args, ' 参数')
    weights_file.close()
    return yolo_weights


def show_weights_shape(weights):
    for i in range(len(weights)):
        sub_weights = weights[i]
        print(np.array(sub_weights).shape)


def preprocess_image(img_path, model_image_size):
    image = Image.open(img_path)

    # 将图像缩放成固定大小608 608
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)

    # 得到图片的像素值，608*608*3
    image_data = np.array(resized_image, dtype='float32')
    # 数据归一化处理
    image_data /= 255.
    # 给数据添加一个新维度，数据量维度，得到1*608*608*3，是适配网络的格式
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    # 返回的image shape是(h, w)格式
    return image, image_data[:, :, :, 0:3], tuple(reversed(image.size))


def readAnchorBoxShape(path):
    with open(path) as f:
        s = f.readline().strip()
        anchors = [float(x) for x in s.split(',')]
        # 5,2
        anchor_box_shape = np.reshape(anchors, [-1, 2])
    return anchor_box_shape


def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def read_class_name(path):
    with open(path) as f:
        lines = f.readlines()
        # 删除一头一尾的空白
        class_name = []
        for x in lines:
            t = x.strip()
            # 忽略空行
            if len(t) > 0:
                class_name.append(t)
    return class_name


# def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
#     font = ImageFont.truetype(font='./font/FiraMono-Medium.otf',
#                               size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
#     thickness = (image.size[0] + image.size[1]) // 300
#     for i, c in reversed(list(enumerate(out_classes))):
#         predicted_class = class_names[c]
#         box = out_boxes[i]
#         score = out_scores[i]
#         label = '{} {:.2f}'.format(predicted_class, score)
#         draw = ImageDraw.Draw(image)
#         label_size = draw.textsize(label, font)
#
#         top, left, bottom, right = box
#         top = max(0, np.floor(top + 0.5).astype('int32'))
#         left = max(0, np.floor(left + 0.5).astype('int32'))
#         bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
#         right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
#         print(label, (left, top), (right, bottom))
#
#         if top - label_size[1] >= 0:
#             text_origin = np.array([left, top - label_size[1]])
#         else:
#             text_origin = np.array([left, top + 1])
#
#         # My kingdom for a good redistributable image drawing library.
#         for i in range(thickness):
#             draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
#         draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
#         draw.text(text_origin, label, fill=(0, 0, 0), font=font)
#         del draw
def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    font = ImageFont.truetype(font='./font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw


def generateXYOffset():
    """
    生成偏移量 19,19,2
    0,0   1,0   2,0   3,0  ...   18,0
    0,1   1,1   2,1   3,1  ...   18,1
                           ...
    0,18  1,18  2,18  3,18 ...   18,18
    由两组数据组成，
    0    1   2   3  ...   w-1
    0    1   2   3  ...   w-1
                    ...

    0    0
    1    1
    2    2
    3    3
    ...  ...  ...
    h-1  h-1

    更一般来讲是0...w-1重复h次 , (0...h-1)T 重复w次
    """
    w = 19
    h = 19

    A = np.tile(np.reshape(np.arange(0, w, dtype='float32'), [1, w]), [h, 1])
    B = np.tile(np.reshape(np.arange(0, h, dtype='float32'), [h, 1]), w)
    A = np.expand_dims(A, axis=-1)
    B = np.expand_dims(B, axis=-1)
    C = np.concatenate([A, B], axis=-1)
    return C
