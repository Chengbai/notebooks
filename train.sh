#!/bin/bash
ulimit -n 2048
if [ -z "$1" ]
then
   python vlm_train.py
else
    python vlm_train.py --checkpoint $1
fi