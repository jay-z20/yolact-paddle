# 训练脚本
python train.py --config=yolact_resnet50_config --batch_size=8



# resume 脚本
python train.py --config=yolact_resnet50_config --batch_size=8  --resume=./weights/yolact_resnet50_43_640000.dpparams --start_iter=640000

# 预测 test-dev 
python eval.py --trained_model yolact_resnet50_54_800000.dpparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True





# python eval.py --trained_model yolact_base_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True

# python eval.py --trained_model yolact_im700_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset  --cuda=True


# python train.py --config=yolact_resnet50_config --batch_size=8 --trained_model=yolact_base_54_800000.pdparams

