"""Miscellaneous utility functions."""

import numpy as np
from PIL import Image

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
    return image, image_data[:, :, :, 0:3], tuple(reversed(image.size))

