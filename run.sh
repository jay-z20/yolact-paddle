# 训练脚本
# validation_epoch 用于设置 valid 测试起始 epoch，默认 35W ，也就是 35w epoch 之后 每 1w 个 epoch 进行 val
# 这里设置为 10000，就是 1w 个 epoch 开始进行 valid
# 因为 valid 写在 保存模型之后，所以 validation_epoch 可设置为 n*1w 
python train.py --config=yolact_resnet50_config --batch_size=8 --validation_epoch=10000



# resume 脚本
# start_iter 用于加载 optimizer 的参数和 epoch，如果 start_iter 不设置，默认 -1 从 0 开始
python train.py --config=yolact_resnet50_config --batch_size=8  --resume=./weights/yolact_resnet50_43_640000.dpparams --start_iter=640000


# valid 脚本
# dataset 默认 coco2017 val2017 4952 个样本
python eval.py --config=yolact_resnet50_config \
--trained_model=weights/yolact_resnet50_43_640000.pth

# 预测 test-dev 
python eval.py --trained_model yolact_resnet50_54_800000.dpparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True

# 打包预测结果
cd results

cp mask_detections.json detections_test-dev2017_yolact_results.json

zip detections_test-dev2017_yolact_results.zip detections_test-dev2017_yolact_results.json


# python eval.py --trained_model yolact_base_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True

# python eval.py --trained_model yolact_im700_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset  --cuda=True


# python train.py --config=yolact_resnet50_config --batch_size=8 --trained_model=yolact_base_54_800000.pdparams

