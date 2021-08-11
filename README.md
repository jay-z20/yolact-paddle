# yolact-paddle

论文：YOLACT: Real-time Instance Segmentation  
使用 Paddle 复现


## 一、内容简介

本项目基于paddlepaddle框架复现YOLACT，YOLACT是一种新的实例分割网络，由于没有使用两阶段方法中的pooling操作使得可以获得无损失的特征信息，并且在大目标的分割场景下性能表现更优秀

论文地址：

https://arxiv.org/pdf/1904.02689.pdf

参考项目：

https://github.com/dbolya/yolact

https://github.com/PaddlePaddle/PaddleDetection

https://github.com/PaddlePaddle/PaddleSeg

## 二、复现精度
**COCO test-dev2017**

| Image Size | Backbone      | FPS  | mAP  |
|:----------:|:-------------:|:----:|:----:|
| 550        | Resnet101-FPN | 6 | 29.8 |
| 700        | Resnet101-FPN | 6 | 31.2 |

## 三、数据集

COCO2017-完整数据集:

https://aistudio.baidu.com/aistudio/datasetdetail/97273

## 四、环境依赖

- 硬件：GPU、CPU

- 框架：
  aistudio 默认 2.1 版本
  Name: paddlepaddle-gpu
  Version: 2.1.2.post101

## 五、快速开始
**训练**
> python train.py --config=yolact_base_config --batch_size=2 --trained_model=yolact_base_54_800000.pdparams

**预测**
> python eval.py --trained_model yolact_base_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True
`result` 文件夹中 `mask_detections.json` 生成结果

