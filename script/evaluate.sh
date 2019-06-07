#!/usr/bin/env bash

rm -r ./_r
rm pckh.csv
python evaluate/crop_market.py
CUDA_VISIBLE_DEVICES=2,3 python2 evaluate/compute_coordinates.py
python evaluate/calPCKH_market.py
python evaluate/cal_apr.py ./_generated