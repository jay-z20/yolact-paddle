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

| Image Size | Backbone      | FPS  | mAP  |download|
|:----------:|:-------------:|:----:|:----:|:---:|
| 550        | Resnet101-FPN | 6 | 29.8 |[百度网盘](https://pan.baidu.com/s/15H0BwHsfFnkjaxD9neiFdA) 提取码: 3g73 |
| 700        | Resnet101-FPN | 6 | 31.2 |[百度网盘](https://pan.baidu.com/s/1fkTqmOXbZOi-TKwJJRwNJg) 提取码: vsv4 |

## 三、数据集

COCO2017-完整数据集:

https://aistudio.baidu.com/aistudio/datasetdetail/97273


image_info_test2017.zip (image_info_test-dev2017.json):

https://cocodataset.org/#download

test2017 (6339.04M):

https://aistudio.baidu.com/aistudio/datasetdetail/12716/0

## 四、环境依赖

- 硬件：GPU、CPU

- 框架：
  
  aistudio 默认 2.1 版本
  
  Name: paddlepaddle-gpu
  
  Version: 2.1.2.post101

## 五、快速开始

**修改训练数据配置文件**

修改配置文件 `data/config.py`

```
'train_images': '/home/aistudio/train2017/', # 修改为训练数据文件夹

'train_info': '/home/aistudio/annotations/instances_train2017.json',  # 修改为 train 据标注
'valid_info': '/home/aistudio/annotations/instances_val2017.json',     # 修改为 val 数据标注

coco2017_testdev_dataset = dataset_base.copy({
    'name': 'COCO 2017 Test-Dev',
    'valid_images': '/home/aistudio/test2017/',                                  # coco test-dev2017 测试数据集文件夹
    'valid_info': '/home/aistudio/annotations/image_info_test-dev2017.json',     # image_info_test-dev2017.json 文件
    'has_gt': False,

    'label_map': COCO_LABEL_MAP
})
```

**预测**
> python eval.py --trained_model yolact_base_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True
`result` 文件夹中 `mask_detections.json` 生成结果

**训练**
> python train.py --config=yolact_base_config --batch_size=2 --trained_model=yolact_base_54_800000.pdparams

**训练 `log`**
`B: Localization Loss   C: Confidence loss  M: Mask loss  S:Semantic segmentation loss`

![image](https://user-images.githubusercontent.com/25956447/129172494-84c6fdb7-16ba-4009-bbaf-ff52be294e0c.png)


## 六、代码结构与详细说明

### 6.1 代码结构

```
├─data                            # 数据加载和配置
   |--config.py                   # 配置文件
├─layers                         # 中间处理过程和 loss
   |--modules
      |--multibox_loss.py         # 训练的 loss
├─logs                            # 训练日志
├─utils                          # 工具包（计时、日志记录、数据增强）
│--backbone.py                   # backbone(resnet 实现)
│--eval.py                        # 评估
│--yolact.py                     # model 实现
│--README.md                   # 中文readme
│--requirement.txt                # 依赖
│--train.py                       # 训练
```

### 6.2 评估流程
> python eval.py --trained_model yolact_im700_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True

在 `coco` 官网提交结果
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.506
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.328
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.121
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.332
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.290
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.608
```
