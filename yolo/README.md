Contents

[Requirements](#requirements)

[Features](#features)

[Preparation](#preparation)

[Training](#training)

[Evaluation](#evaluation)

[Benchmarking the speed of network](#benchmarking-the-speed-of-network)

---

### Requirements
- python 3.6
- pytorch 0.4.0
### Features
- Include both Yolov2 and Yolov3
- Good performance

  |544x544 |VOC2007 Test(mAP)|Time per forward<br/>(batch size = 1)|
  | :-: | :-:|:-:|
  | Yolov2  | 77.6% |11.5ms|
  | Yolov3  | 79.6% |23.1ms|
  
  The models are trained from pretrained weights on imagenet **with this implementation**.

- Train as fast as [darknet](https://github.com/pjreddie/darknet)
- A lot of efficient backbones on hand

  Like tiny yolov2, tiny yolov3, mobilenet, mobilenetv2, shufflenet(g2), shufflenetv2(1x), squeezenext(1.0-SqNxt-23v5), light xception, xception etc. 

  Check folder `vedanet/network/backbone` for details.
  
  |416x416 |VOC2007 Test(mAP)| Time per forward<br/>(batch size = 1)|
  | :-: | :-:| :-: |
  | TinyYolov2  | 57.5% | 2.4ms|
  | TinyYolov3  | 61.3% | 2.3ms|

  The models are **trained from scratch with this implementation**.
---

### Preparation
##### 1) Code
`git clone https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet`

`cd ObjectDetection-OneStageDet/yolo`

`yolo_root=$(pwd)`

`cd ${yolo_root}/utils/test`

`make -j32`

##### 2) Data
`wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar`

`wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar`

`wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar`

`tar xf VOCtrainval_11-May-2012.tar`

`tar xf VOCtrainval_06-Nov-2007.tar`

`tar xf VOCtest_06-Nov-2007.tar`

`cd VOCdevkit`

`VOCdevkit_root=$(pwd)`

There will now be a VOCdevkit subdirectory with all the VOC training data in it.

`mkdir ${VOCdevkit_root}/onedet_cache`

`cd ${yolo_root}`

open examples/labels.py, let the variable `ROOT` point to `${VOCdevkit_root}`

`python examples/labels.py` 

open cfgs/yolov2.yml, let the `data_root_dir` point to `${VOCdevkit_root}/onedet_cache`

open cfgs/yolov3.yml, let the `data_root_dir` point to `${VOCdevkit_root}/onedet_cache`

##### 3) weights
Download model weights from [baidudrive](https://pan.baidu.com/s/1a3Z5IUylBs6rI-GYg3RGbw) or [googledrive](https://drive.google.com/open?id=1nW3u35_5b0ILs2u9TOQ5Nubjx8-1ewwc).

Or downlowd [darknet19_448.conv.23](https://pjreddie.com/media/files/darknet19_448.conv.23) and [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74) from darknet website:

`wget https://pjreddie.com/media/files/darknet19_448.conv.23`

`wget https://pjreddie.com/media/files/darknet53.conv.74`

Then, move all the model weights to `${yolo_root}/weights` directory.

---

### Training
`cd ${yolo_root}`

##### 1) Yolov2

1.1) open cfgs/yolov2.yml, let the `weights` of `train` block point to the pretrain weights

1.2) open cfgs/yolov2.yml, let the `gpus` of `train` block point to an available gpu id

1.3) If you want to print log onto screen, make the `stdout` of `train` block `True` in cfgs/yolov2.yml

1.4) run

`python examples/train.py Yolov2`

##### 2) Yolov3
2.1) open cfgs/yolov3.yml, let the `weights` of `train` block point to the pretrain weights

2.2) open cfgs/yolov3.yml, let the `gpus`  of `train` block point to an available gpu id

2.3) If you want to print log onto screen, make the `stdout` of `train` block `True` in cfgs/yolov3.yml

2.4) run

`python examples/train.py Yolov3`

##### 3) Results
The logs and weights will be in `${yolo_root}/outputs`.

##### 4) Other models
There are many other models like tiny yolov2, tiny yolov3, mobilenet, mobilenetv2, shufflenet(g2), shufflenetv2(1x), squeezenext(1.0-SqNxt-23v5), light xception, xception etc. You can try these like `1) Yolov2` part.

---

### Evaluation
`cd ${yolo_root}`

##### 1) Yolov2
1.1) open cfgs/yolov2.yml, let the `gpus` of `test` block point to an available gpu id

1.2) run

`python examples/test.py Yolov2`

##### 2) Yolov3
2.1) open cfgs/yolov3.yml, let the `gpus` of `test` block point to an available gpu id

2.2) run

`python examples/test.py Yolov3`

##### 3) Results
The output bbox will be in `${yolo_root}/results`,  every line of the file in   `${yolo_root}/results` has a format like `img_name confidence xmin ymin xmax ymax`

##### 4) Other models
There are many other models like tiny yolov2, tiny yolov3, mobilenet, mobilenetv2, shufflenet(g2), shufflenetv2(1x), squeezenext(1.0-SqNxt-23v5), light xception, xception etc. You can try these like `1) Yolov2` part.

---

### Benchmarking the speed of network
`cd ${yolo_root}`

##### 1) Yolov2
1.1) open cfgs/yolov2.yml, let the `gpus` of `speed` block point to an available gpu id

1.2) run

`python examples/speed.py Yolov2`

##### 2) Yolov3
2.1) open cfgs/yolov3.yml, let the `gpus` of `speed` block point to an available gpu id

2.2) run

`python examples/speed.py Yolov3`

##### 3) Tiny Yolov2
3.1) open cfgs/tiny_yolov2.yml, let the `gpus` of `speed` block point to an available gpu id

3.2) run

`python examples/speed.py TinyYolov2`

##### 4) Tiny Yolov3
4.1) open cfgs/tiny_yolov3.yml, let the `gpus` of `speed` block point to an available gpu id

4.2) run

`python examples/speed.py TinyYolov3`

##### 5) Mobilenet
5.1) open cfgs/region_mobilenet.yml, let the `gpus` of `speed` block point to an available gpu id

5.2) run

`python examples/speed.py RegionMobilenet`

##### 6) Other backbones with region loss
You can try these like `5) Mobilenet` part.

---

### Credits
I got a lot of code from [lightnet](https://gitlab.com/EAVISE/lightnet), thanks to EAVISE.
