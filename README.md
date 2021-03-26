# yolov2 by tf2


---

使用 **tensorflow2.0** 实现的 **YOLO**_(you only look once)_ v2算法

_[YOLO](https://pjreddie.com/darknet/yolo/) 是一个实时对象检测系统_

---

## 算法效果

## 算法原理

---

#### 网络的输出

---

网络最终输出一个`(19,19,425)`的张量，我们可以将其转换成`(19,19,5,85)`的张量，
其中`[i,j,a,:]`表示在第`[i,j]`单元格的第`a`个`anchor-box`的预测结果，由`4`部分组成

1. `[0:2]` 存储了对象中心在单元格内的位置`(x,y)`坐标
   - 经过`sigmoid`函数映射，范围限制在`0～1`
2. `[2:4]` 存储了对象的宽高`(w,h)`
   - 经过`exp`函数映射得到关于对应`anchor-box`的宽高系数，必须大于`0`
3. `[4:5]` 存储了该`anchor-box`包含对象的置信度(概率)
   - 经过`sigmoid`函数映射，范围限制在`0～1`
4. `[5:85]` 存储了该`anchor-box`包含的对象关于`80`个类别的概率分布
   - 经过`softmax`归一化，范围限制在`0～1`，总和为`1`

注意网络输出值并没有经过上面所描述的映射，

---

#### 人工标签

维度是`(K,5)`的张量，`K`为图片中包含的对象数量，每个对象由`3`部分决定

1. `[0:2]` 存储了对象中心在整张图片的相对位置`(lx,ly)`，范围在`0～1`
2. `[2:4]` 存储了对象的宽高`(lw,lh)`，范围在`0～1`
3. `[4:5]` 存储了对象的类别`class`，范围在`0～79`

#### 人工标签转换

人工标签的维度是`(K,5)`，显然与网络输出维度不符合，为了计算网络`loss`，我们将
人工标签转换得到`(19,19,5)`维度的张量`detectors_mask`和
`(19,19,5,5)`维度的张量`matching_true_boxes`

---

1. `detectors_mask` 

元素为`bool`类型，`detectors_mask[i,j,a]`为`True`，
则表明在第`[i,j]`单元格的第`a`个`anchor-box`的存在对象

---

2. `matching_true_boxes` 
   
若`detectors_mask[i,j,a]`为`True`,
则`matching_true_boxes[i,j,a,:]`存储了对象信息，由`3`部分决定

   1. `[0:2]` 存储了对象中心在单元格内的位置`(x,y)`坐标
      - 变换规则，`(lx,ly) * (19,19) = (19lx ,19ly) = (u,v)`，
      `(x,y) = (u,v) - floor((u,v))`
   2. `[2:4]` 存储了对象的宽高`(w,h)`，与网络输出的宽高含义一致
      - 变换规则，`(lw,lh) ⊙ (19,19) = (19lw ,19lh) = (p,q)`，
      `(w,h) = log((p,q) / (anchorW,anchorH))`
   3. `[4:5]` 存储了对象的类别`class`，范围在`0～79`
      - 可以使用`one_hot`函数转换成相应的`softmax`标签
    
---

#### Loss函数计算


<img width="623" alt="loss" src="https://user-images.githubusercontent.com/19931702/112642948-c4942a00-8e7e-11eb-929c-a6b39623e536.png">


---

上述表达式中`detectors_mask` 前面提到过


1. `object_detections`

`object_detections(i,j,a)` 为`True`，表示该位置预测框与一个真实对象框很吻合(具体是**IOU**>_threshold=0.6_)
，此时即使该位置本不应存在对象及`detectors_mask(i,j,a)=False`也不做`noobj`置信度`loss`计算。
一方面为了减少负例，另外一方面是网络在本不应该包含的位置输出了较为吻合的锚框，
仍然可以通过非最大值抑制算法滤去。这通常发生在一个格子中的多个锚框检测同一个对象的时候。

---

2. 有`4`个权重系数，这里实现上分别取值为`lambda_obj=5`、`lambda_noobj=1`、`lambda_coord=1`、`lambda_class=1`

3. 字母`N`表示类别的数量，`yolov2`系统中是`80`

---
