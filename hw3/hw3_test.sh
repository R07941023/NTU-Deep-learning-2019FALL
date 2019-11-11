#!/bin/bash 
wget https://www.dropbox.com/s/5jouiri2mdodskh/233_model.pth?dl=1 -O ./233_model.pth

python3 hw3_test.py $1 $2
