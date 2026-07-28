# yolov2 by tf2


---

**YOLO**_(you only look once)_ v2 algorithm implemented using **tensorflow2.0**

_[YOLO](https://pjreddie.com/darknet/yolo/) is a real-time object detection system._

This project implements all the details of the system, including model construction, Loss functions, object detection, and how to **train from scratch**, allowing for extension to training custom object detection systems.

---

## Algorithm Effects

Detection results

<img alt="" src="https://user-images.githubusercontent.com/19931702/113099260-447a1580-922c-11eb-8d5a-8707fb75b16e.png">
<img alt="" src="https://user-images.githubusercontent.com/19931702/113131897-5c639080-9250-11eb-9bac-c527abb3c55c.png">
<img alt="" src="https://user-images.githubusercontent.com/19931702/113132879-9a14e900-9251-11eb-80b5-2f19ad441275.jpg">




---

## How to Use

#### Environment Requirements
1. Anconda        2020.11
2. Conda          4.9.2
3. Python         3.7.10
4. Tensorflow     2.0.0

---

1. Detect objects in images

```
python test_tect.py
```

2. Save yolo_v2 model

```
python save_model.py
```

3. Detect images using the saved yolo_v2 model
```
python load_model_and_test_detect.py
```

4. Train the network from scratch (train an overfitted model on a training set of 2 samples)
```
python train_model_overfit.py
```

5. Perform object detection using the previously trained overfitted model
```
python load_overfit_model_and_test_detect.py
```

## Algorithm Principles

---

#### 1. Network Architecture

<img width="1099" alt="" src="https://user-images.githubusercontent.com/19931702/113086704-c2322700-9214-11eb-90b8-dcd510d48bc5.png">




#### 2. Network Output

---

The network ultimately outputs a tensor of shape `(19,19,425)`, which we can convert into a tensor of shape `(19,19,5,85)`.
Among these, `[i,j,a,:]` represents the prediction result for the `a`-th `anchor-box` in the `[i,j]` cell, consisting of `4` parts:

1. `[0:2]` stores the position `(x,y)` coordinates of the object center within the cell.
   - Mapped via `sigmoid` function, range limited to `0~1`.
2. `[2:4]` stores the width and height `(w,h)` of the object.
   - Mapped via `exp` function to obtain the width/height coefficients relative to the corresponding `anchor-box`; must be greater than `0`.
3. `[4:5]` stores the confidence (probability) that the `anchor-box` contains an object.
   - Mapped via `sigmoid` function, range limited to `0~1`.
4. `[5:85]` stores the probability distribution of the object within the `anchor-box` across `80` categories.
   - Normalized via `softmax`, range limited to `0~1`, summing to `1`.

_Note: The network output values have not undergone the mappings described above; that is, the network is learning the input values for these functions._

---

#### 3. Ground Truth Labels

The dimension is a tensor of `(K,5)`, where `K` is the number of objects in the image. Each object is determined by `3` parts:

1. `[0:2]` stores the relative position `(lx,ly)` of the object center in the whole image, range `0~1`.
2. `[2:4]` stores the width and height `(lw,lh)` of the object, range `0~1`.
3. `[4:5]` stores the object `class`, range `0~79`.

#### 4. Ground Truth Label Conversion

The dimension of the labels is `(K,5)`, which obviously does not match the network output dimensions. To calculate the network `loss`, we convert the labels into a `(19,19,5)` dimensional tensor `detectors_mask` and a `(19,19,5,5)` dimensional tensor `matching_true_boxes`.

---

1. `detectors_mask` 

Elements are of `bool` type. If `detectors_mask[i,j,a]` is `True`, it indicates that an object exists in the `a`-th `anchor-box` of the `[i,j]` cell.

---

2. `matching_true_boxes` 
   
If `detectors_mask[i,j,a]` is `True`, then `matching_true_boxes[i,j,a,:]` stores the object information, determined by `3` parts:

   1. `[0:2]` stores the position `(x,y)` coordinates of the object center within the cell.
      - Transformation rule: `(lx,ly) * (19,19) = (19lx, 19ly) = (u,v)`, then `(x,y) = (u,v) - floor((u,v))`.
   2. `[2:4]` stores the width and height `(w,h)` of the object, consistent with the meaning of the network output.
      - Transformation rule: `(lw,lh) ⊙ (19,19) = (19lw, 19lh) = (p,q)`, then `(w,h) = log((p,q) / (anchorW,anchorH))`.
   3. `[4:5]` stores the object `class`, range `0~79`.
      - Can be converted to corresponding `softmax` labels using the `one_hot` function.
    
---

#### 5. Loss Function Calculation


<img width="623" alt="loss" src="https://user-images.githubusercontent.com/19931702/112642948-c4942a00-8e7e-11eb-929c-a6b39623e536.png">


---

Total `Loss` = Detection `Loss` + Classification `Loss` + Localization `Loss`, all using square error `Loss`. `detectors_mask` was mentioned previously.

1. `object_detections`

If `object_detections(i,j,a)` is `True`, it means the predicted box at this position fits a ground truth object box very well (specifically **IOU** > _threshold=0.6_). In this case, even if an object should not actually exist at this location (`detectors_mask(i,j,a)=False`), the `no-obj loss` is not calculated. The reasons are as follows:
    
- During network training, ground truth objects are assigned to a pair of `(grid cell, anchor-box)`. However, a cell contains multiple anchor-boxes. In reality, if an object shape is close to multiple `anchor-boxes` (`IOU` is close), it is reasonable for the object to be assigned to any of those `anchor-boxes`. Therefore, it is also reasonable for the network to output predicted boxes in multiple locations, even though we only select one `(grid cell, anchor-box)` with the optimal `IOU` for labeling. Thus, we can relax the requirements: if the network says there is an object near the ground truth position and the predicted box matches the ground truth box well, we neither punish nor encourage the network, remaining neutral. Furthermore, these redundant predictions can be filtered out by the Non-Maximum Suppression (NMS) algorithm.
- On the other hand, if we are very strict and apply `no-obj loss` punishment to these areas, there will be too many negative examples. In one image, the network predicts `19*19*5=1805` boxes, while typically ground truth objects are fewer than `100`. This means there would be `1705` negative examples, which could lead the network to eventually just learn that there are no objects at any given position.

---

2. There are `4` weight coefficients; in this implementation, they are `lambda_obj=5`, `lambda_noobj=1`, `lambda_coord=1`, `lambda_class=1`.

3. The letter `N` represents the number of categories, which is `80` in the `yolov2` system.

---

#### 6. Reducing Loss with Stochastic Gradient Descent

<img width="326" alt="0" src="https://user-images.githubusercontent.com/19931702/112708294-16bf6480-8eec-11eb-9aa2-921b7315ce20.png">

<img width="193" alt="1" src="https://user-images.githubusercontent.com/19931702/112708320-3a82aa80-8eec-11eb-998d-910442ba4f74.png">

<img width="486" alt="2" src="https://user-images.githubusercontent.com/19931702/112708364-8897ae00-8eec-11eb-975a-79f1be00f910.png">

<img width="336" alt="3" src="https://user-images.githubusercontent.com/19931702/112708396-c1d01e00-8eec-11eb-8332-6c2dfb9f91f3.png">

<img width="625" alt="4" src="https://user-images.githubusercontent.com/19931702/112708408-dca29280-8eec-11eb-8f8f-c834180008d7.png">

<img width="622" alt="5" src="https://user-images.githubusercontent.com/19931702/112708586-c77a3380-8eed-11eb-828a-7bd3f2e9c970.png">

<img width="624" alt="6" src="https://user-images.githubusercontent.com/19931702/112708595-cfd26e80-8eed-11eb-8941-1e6ab651f603.png">

<img width="482" alt="7" src="https://user-images.githubusercontent.com/19931702/112708684-6868ee80-8eee-11eb-8b52-088ee7f07c2f.png">

<img width="624" alt="8" src="https://user-images.githubusercontent.com/19931702/112708689-774fa100-8eee-11eb-8871-004d08b42dc1.png">

<img width="424" alt="9" src="https://user-images.githubusercontent.com/19931702/112708691-7a4a9180-8eee-11eb-90b2-906e3a6992dd.png">

<img width="533" alt="10" src="https://user-images.githubusercontent.com/19931702/112708779-f349e900-8eee-11eb-8eb7-c07633fc2fdf.png">

---

For the implementation of how the backpropagation algorithm calculates parameter gradients, please refer to my other two projects:

- [CNN](https://github.com/970814/convolutionNerualNetwork)
- [MLP](https://github.com/970814/fcBpNerualNetwork)

If using the `tensorflow` framework, gradient calculation is handled automatically by the framework, meaning we only need to implement the forward propagation algorithm and the loss function. This is a huge advantage of using a framework to implement models.

---

#### 7. Performing Prediction

The input image is a tensor of `(608,608,3)`. After passing through our trained `D-CNN` (a deep convolutional network with `23` convolutional layers), the network ultimately outputs a tensor of `(19,19,425)`, converted into a tensor of `(19,19,5,85)`, which means `(19,19,5)` object detection results.
Each detection result has the confidence `conf = [4:5]` that an object exists at that location, and a probability distribution `prob=[5:85]` for `80` categories.
Therefore, the score for each category is `score = conf * prob`. We then take the category with the highest score as the detection result.
Since there are `(19,19,5)` object detection results, there are `19*19*5=1805` scores. We filter out predictions with low scores (`score < threshold=0.6`).
The remaining results are detections with high scores. However, because during training *we did not punish detection results near objects containing objects*, the network may detect the same object in multiple locations.
Therefore, we next apply the Non-Maximum Suppression (NMS) algorithm to each category to obtain the final object detection results. The principles and implementation of this part can be found in the `convert_filter_and_non_max_suppression` function in the `utils.py` file.

---

### Reference Materials

- [YAD2K](https://github.com/allanzelener/YAD2K)
- [Simple and Brutal TensorFlow2](https://tf.wiki/zh_hans/basic/models.html)
- [Andrew Ng Deep Learning YOLO Algorithm](https://www.bilibili.com/video/BV1F4411y7o7?p=31)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) 
- [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
- [Final Layers and Loss Function](https://medium.com/oracledevs/final-layers-and-loss-functions-of-single-stage-detectors-part-1-4abbfa9aa71c)
- [YOLO Loss Details](https://www.jianshu.com/p/e6582dfa6bb3?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation)
- [Yolov2 Loss Function Details](https://zhuanlan.zhihu.com/p/56079893)
- [BatchNormalization Pitfalls Experience](https://zhuanlan.zhihu.com/p/64310188)
