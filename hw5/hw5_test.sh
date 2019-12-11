#!/bin/bash 
wget https://www.dropbox.com/s/dbskk15i7amazsu/good.pth?dl=1 -O ./good.pth
python3 hw5_test.py $1 $2
