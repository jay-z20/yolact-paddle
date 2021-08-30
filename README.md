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

| Image Size | Backbone      | mAP  |download|
|:----------:|:-------------:|:----:|:---:|
| 550        | Resnet50-FPN | 28.9 |[百度网盘](https://pan.baidu.com/s/1vRyiI__ewKWrI0rdIXkYCg) 提取码: 6jjs |

## 三、数据集

COCO2017-完整数据集:

https://aistudio.baidu.com/aistudio/datasetdetail/97273


## 四、环境依赖

> pip install -r requirments.txt

- 硬件：GPU、CPU

- 框架：
  
  aistudio 默认 2.1 版本
  
  Name: paddlepaddle-gpu
  
  Version: 2.1.2.post101

## 五、快速开始

**修改训练数据配置文件**

修改配置文件 `data/config.py`

```
dataset_base = Config({
    'name': 'Base Dataset',

    # Training images and annotations
    'train_images': '/home/aistudio/train2017/', # 修改为训练数据文件夹
    'train_info':   'path_to_annotation_file',

    # Validation images and annotations.
    'valid_images': '/home/aistudio/val2017/',    # 修改为验证数据文件夹
    'valid_info':   'path_to_annotation_file',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    # A list of names for each of you classes.
    'class_names': COCO_CLASSES,

    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None
})

coco2017_dataset = dataset_base.copy({
    'name': 'COCO 2017',
    
    'train_info': '/home/aistudio/annotations/instances_train2017.json', # 修改为 train 据标注
    'valid_info': '/home/aistudio/annotations/instances_val2017.json',   # 修改为 val 数据标注

    'label_map': COCO_LABEL_MAP
})

coco2017_testdev_dataset = dataset_base.copy({
    'name': 'COCO 2017 Test-Dev',
    'valid_images': '/home/aistudio/test2017/',                                  # coco test-dev2017 测试数据集文件夹
    'valid_info': '/home/aistudio/annotations/image_info_test-dev2017.json',     # image_info_test-dev2017.json 文件
    'has_gt': False,

    'label_map': COCO_LABEL_MAP
})
```

**预测**

> python eval.py --trained_model yolact_resnet50_54_800000.dpparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True

> `result` 文件夹中 `mask_detections.json` 生成结果

**训练**
> python train.py --config=yolact_resnet50_config --batch_size=8


## 六、代码结构与详细说明

### 6.1 代码结构

```
├─data                            # 数据加载和配置
   |--config.py                   # 配置文件
├─layers                          # 中间处理过程和 loss
   |--modules
      |--multibox_loss.py         # 训练的 loss
├─logs                            # 训练日志 每 10 个 epoch 打印
├─utils                           # 工具包（计时、日志记录、数据增强）
|--weights                        # 模型保存目录
│--backbone.py                    # backbone(resnet 实现)
│--eval.py                        # 预测和评估
│--yolact.py                      # model 实现
│--README.md                      # 中文readme
│--requirement.txt                # 依赖
│--train.py                       # 训练
```

### 6.2 评估流程
> 下载 yolact_resnet50_54_800000.dpparams 并保存到 eval.py 相同目录下
> 
> python eval.py --trained_model  yolact_resnet50_54_800000.dpparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True
> 
> cd results
> 
> cp mask_detections.json detections_test-dev2017_yolact_results.json
> 
> zip detections_test-dev2017_yolact_results.zip detections_test-dev2017_yolact_results.json

在 `coco` 官网提交结果
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.289
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.477
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.301
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.093
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.301
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.460
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.273
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.395
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.407
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.439
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
```
