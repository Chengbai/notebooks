#!/bin/bash
ulimit -n 2048
if [ -z "$1" ]
then
   python exp_att_val_transformer/vlm_att_val_transformer_train.py
else
    python exp_att_val_transformer/vlm_att_val_transformer_train.py --checkpoint $1
fi