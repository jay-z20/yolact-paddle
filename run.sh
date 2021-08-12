python eval.py --trained_model yolact_base_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True

python eval.py --trained_model yolact_im700_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset  --cuda=True

python train.py --config=yolact_base_config --batch_size=8 --trained_model=yolact_base_54_800000.pdparams
