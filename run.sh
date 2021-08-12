python eval.py --trained_model yolact_base_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset --cuda=True
python eval.py --trained_model yolact_im700_54_800000.pdparams --output_coco_json  --dataset=coco2017_testdev_dataset

python train.py --config=yolact_base_config --batch_size=8 --trained_model=yolact_base_54_800000.pdparams

python eval.py --trained_model yolact_base_54_800000.pdparams --output_coco_json --images=/home/aistudio/images/:/home/aistudio/res/
"--images","./images/:./outputs/"
python run_coco_eval.py