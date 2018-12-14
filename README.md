# MobileNetv2-SSDLite
Caffe implementation of SSD detection on MobileNetv2, converted from tensorflow.  

This repo. **provides deploy.caffemodel and deploy.prototxt trained via COCO dataset** that work fine for object detection task. If you want to remake your caffemodel and prototxt via own dataset then **read "how to make new caffemodel and prototxt"** section.  

### Prerequisites
Tensorflow and Caffe version [SSD](https://github.com/weiliu89/caffe) is properly installed on your computer.

### Usage
Detection Demo bellow,
```
    $ python demo_caffe.py
```

### How to make new caffemodel and prototxt

#### **Notice!: As of now ssd/ version is working fine, but ssdlite/ version bellow is not working correctly.**

0. Firstly you should download the original model from [tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).  
```
    $ pip install tensorflow==1.5
``` 

1. Use gen_model.py to generate the train.prototxt and deploy.prototxt (or use the default prototxt).  
coco/labelmap_coco.prototxt includes **10** name: 'N/A' and **10** display_name: 'N/A' and background, so number of **classes is 91(=80+1+10)**. 
```
    // -c 91 means category number of coco included 'N/A'(==10 items) and background.
    
    // for ssd
    $ python ssd/gen_model.py -s deploy -c 91 >deploy.prototxt
    
    // for ssdlite
    $python ssdlite/gen_model.py -s deploy -c 91 >deploy.prototxt
```
Check **generated .prototxt** directory.
    
2. Use dump_tensorflow_weights.py to dump the weights of conv layer and batchnorm layer.  
```
    // for ssd
    $ wget  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    $ tar xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    $ python ssd/dump_tensorflow_weights.py
    
    // for ssdlite
    $ wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
    $ tar xzf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
    $ python ssdlite/dump_tensorflow_weights.py
```
Check **generated contents of ./output/** directory.

3. Use load_caffe_weights.py to load the dumped weights to deploy.caffemodel.  
```
    // for ssd
    $ python ssd/load_caffe_weigts.py

    // for ssdlite
    $ python ssdlite/load_caffe_weigts.py
```
Check **generated deploy.caffemodel**.

4. Use the code in src to accelerate your training if you have a cudnn7, or add "engine: CAFFE" to your depthwise convolution layer to solve the memory issue.
5. The original tensorflow model is trained on MSCOCO dataset, maybe you need deploy.caffemodel for VOC dataset, use coco2voc.py to get deploy_voc.caffemodel.

### Attempt Demo Scripts
Try Object Detection Demo to check **generated deploy.prototxt and deploy.caffemodel**.

    // modify demo script
    $ vi demo_caffe.py
    net_file    = 'deploy.prototxt'
    caffe_model = 'deploy.caffemodel'
    $ python demo_caffe.py

### Train your own dataset
1. Generate the trainval_lmdb and test_lmdb from your dataset.
2. Write a labelmap.prototxt
3. Use gen_model.py to generate some prototxt files, replace the "CLASS_NUM" with class number of your own dataset.
```
python gen_model.py -s train -c CLASS_NUM >train.prototxt
python gen_model.py -s test -c CLASS_NUM >test.prototxt
python gen_model.py -s deploy -c CLASS_NUM >deploy.prototxt
```
4. Copy coco/solver_train.prototxt and coco/train.sh to your project and start training.

### Note
There are some differences between caffe and tensorflow implementation:
1. The padding method 'SAME' in tensorflow sometimes use the [0, 0, 1, 1] paddings, means that top=0, left=0, bottom=1, right=1 padding. In caffe, there is no parameters can be used to do that kind of padding.
2. MobileNet on Tensorflow use ReLU6 layer y = min(max(x, 0), 6), but caffe has no ReLU6 layer. Replace ReLU6 with ReLU cause a bit accuracy drop in ssd-mobilenetv2, but very large drop in ssdlite-mobilenetv2. There is a ReLU6 layer implementation in my fork of [ssd](https://github.com/chuanqi305/ssd).


