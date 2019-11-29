#!/bin/bash 
wget https://www.dropbox.com/s/7o2gacyu2ohcxz3/100_model.pth?dl=1 -O ./100_model.pth
python3 hw4_train.py $1 $2
