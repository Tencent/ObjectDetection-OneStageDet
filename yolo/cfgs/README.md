GPU: Tesla V100
Deep learning library: PyTorch 0.4.0
Input size: 544x544
Batch size: 1
Test iterations: 200
Time: 2018-12-29 16:16

##################################

tiny yolov3 < tiny yolov2 < RegionMobilenetv2 < RegionMobilenet < RegionShufflenet & RegionShufflenetv2 
    < RegionLightXception < yolov2 < RegionSqueezenext < RegionXception < Yolov3

yolov2: 11.5 ms/iter
yolov3: 23.1 ms/iter

RegionMobilenet: 6.3 ms/iter
RegionMobilenetv2: 5.9 ms/iter

RegionShufflenet: 7.2 ms/iter
RegionShufflenetv2: 7.4 ms/iter

RegionXception: 22.7 ms/iter
RegionLightXception: 11.0 ms/iter

RegionSqueezenext: 13.8 ms/iter

tiny yolov2: 3.7 ms/iter 
tiny yolov3: 3.4 ms/iter 
