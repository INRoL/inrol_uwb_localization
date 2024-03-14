#!/bin/bash
python3 python/create_colmap_prj.py \
  --imgdir example/anchor_self_calibration1/parsed_dataset/img \
  --param example/config/parameter.yaml \
  --outdir example/anchor_self_calibration1/02-1_create_colmap_prj
