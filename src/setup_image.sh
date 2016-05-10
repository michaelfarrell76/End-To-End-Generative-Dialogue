#!/usr/bin/env bash
# 
# Install script for parallel that uses local file init.lua
#


scp -i ~/.ssh/gcloud-sshkey gcloud_startup.sh $USERNAME@$EXTERNAL_IP:~/

echo "bash startup.sh" | ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey $USERNAME@$EXTERNAL_IP

scp -r -i ~/.ssh/gcloud-sshkey ../data.zip $USERNAME@$EXTERNAL_IP:~/End-To-End-Generative-Dialogue/
scp -i ~/.ssh/gcloud-sshkey ~/installs/lua---parallel/init.lua $USERNAME@$EXTERNAL_IP:~/lua---parallel/

echo "bash gcloud_startup.sh; cd lua---parallel; luarocks make; cd ../End-To-End-Generative-Dialogue; unzip data.zip; cd src; python preprocess.py" | ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey $USERNAME@$EXTERNAL_IP
