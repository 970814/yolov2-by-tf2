import tensorflow as tf
import numpy as np


def loss_function(predictions, labels):
    """
    predictions: D-CNN 的输出，维度是(Batch, 19, 19, 425)
    labels:      人工标签，    维度是(Batch, ?, 5)
    """
    # 首先将predictions转换成(Batch, 19, 19, 5, 85)
    predictions = tf.reshape(predictions, [-1, 19, 19, 5, 85])
    print(np.shape(predictions))
    print("m: ", len(labels))
    '''
    为了计算loss，需要将labels转换成(Batch, 19, 19, 5, 85)维度
    创建bool 类型的 (Batch, 19, 19, 5) detectors_mask，为True的位置表示存在对象，对象的(x,y,w,h,class)存储在相应labels2的位置，
    '''
    detectors_mask, labels2 = convert(labels)
    print(np.shape(detectors_mask))
    print(np.shape(labels2))
    return


def convert(labels):
    # labels   人工标签，    维度是(Batch, ?, 5)

    # 样本数量
    m = len(labels)
    # 创建bool 类型的 (Batch, 19, 19, 5) detectors_mask，为True的位置表示存在对象
    detectors_mask = np.zeros([m, 19, 19, 5])
    # 创建(Batch, 19, 19, 5, 5) labels2，detectors_mask[b,i,j,a]为True的位置,表示labels2[b,i,j,a]存储了目标的(x,y,w,h,class)信息
    labels2 = np.zeros([m, 19, 19, 5, 5])

    print(np.shape(detectors_mask))
    print(np.shape(labels2))

    n = -1
    # 把labels中的值填入到 labels2和detectors_mask之中
    # 遍历每张图片
    for label in labels:
        n += 1  # 当前处理的样本索引
        # label 维度为(K,5)
        # 遍历每张图片里面的目标
        for obj in label:
            # obj 维度为(5,)
            # 获取每个目标的位置类别信息
            xy = obj[0:2]  # (2,),范围0～1
            wh = obj[2:4]  # (2,),范围0～1
            table_size = [19, 19]  # (2,)

            # 转换单位，单位是单元格大小
            xy = np.multiply(xy, table_size)  # (2,),范围0～19
            wh = np.multiply(wh, table_size)  # (2,),范围0～19
            # 计算目标中心所在的单元格位置
            i = tf.cast(tf.floor(xy[1]), 'int32')  # y 方向的偏移量
            j = tf.cast(tf.floor(xy[0]), 'int32')  # x 方向的偏移量

            # 接下来找出与目标形状最相似的anchor-box
            # 每个单元格有5个锚框5个shape，维度为 (5,2)
            # anchor_box_shape = readAnchorBoxShape('./model_data/yolo_anchors.txt')
            # 单位也是单元格大小 维度为(5,2)
            anchor_box_shapes = np.array(
                [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778],
                 [9.77052, 9.16828]])
            # 计算目标与每个anchor-box的交并比
            # 1,2
            obj_wh = np.expand_dims(wh, axis=0)
            obj_xy_max = np.divide(obj_wh, 2)
            obj_xy_min = -obj_xy_max
            # 5,2
            anchor_box_xy_max = np.divide(anchor_box_shapes, 2)
            anchor_box_xy_min = -anchor_box_xy_max

            # 计算交集
            intersection_min = np.maximum(obj_xy_min, anchor_box_xy_min)
            intersection_max = np.minimum(obj_xy_max, anchor_box_xy_max)
            intersection_wh = np.maximum((intersection_max - intersection_min), 0)
            # s = w * h, 维度为(5,)
            intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
            # 计算并集
            # 维度为(1,)
            s1 = obj_wh[:, 0] * obj_wh[:, 1]
            # 维度为(5,)
            s2 = anchor_box_shapes[:, 0] * anchor_box_shapes[:, 1]
            # (1,) + (5,) - (5,) = (5,)
            union = s1 + s2 - intersection
            # 计算iou
            iou = intersection / union
            print(iou)
            # 得出形状最符合的anchor-box索引
            a = np.argmax(iou)

            if detectors_mask[n, i, j, a]:
                raise Exception('人工标注数据有误，同一个位置存在多次标记')

            print(n, i, j, a, ': 位置存在对象')
            # 标记该位置存在对象
            detectors_mask[n, i, j, a] = True
            # 在相应的位置存储目标位置类别信息
            labels2[n, i, j, a] = [
                xy[0] - j,  # x  相对单元格的位置
                xy[1] - i,  # y  相对单元格的位置
                # 我们没有让网络直接学习目标的宽高，而是对应的anchor-box高宽系数的对数
                np.log(wh[0] / anchor_box_shapes[a][0]),  # w  让网络学习anchor-box宽度系数的对数
                np.log(wh[1] / anchor_box_shapes[a][1]),  # h  让网络学习anchor-box高度系数的对数
                obj[4]
            ]

    return detectors_mask, labels2

# 4 14 3
# 9 7 4
# 12 5 4
# 12 8 4
