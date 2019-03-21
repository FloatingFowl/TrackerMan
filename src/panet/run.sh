#!/usr/bin/env bash
python tools/infer_simple.py --dataset coco2017 --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml --load_ckpt ./panet_mask_step179999.pth --image_dir=../images --output_dir=../output --ndct_dump=../ndct.pkl
