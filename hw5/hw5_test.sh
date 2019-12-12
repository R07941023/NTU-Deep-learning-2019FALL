#!/bin/bash 

wget https://www.dropbox.com/s/lx8p261pil8kg5i/embed?dl=1 -O ./embed
wget https://www.dropbox.com/s/dbskk15i7amazsu/good.pth?dl=1 -O ./good.pth
python3 hw5_test.py $1 $2
