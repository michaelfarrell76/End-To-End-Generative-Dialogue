IP_ADDR=130.211.160.115

scp -i ~/.ssh/gcloud-sshkey startup.sh michaelfarrell@$IP_ADDR:~/

echo "bash startup.sh" | ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey michaelfarrell@$IP_ADDR

scp -r -i ~/.ssh/gcloud-sshkey ../data.zip michaelfarrell@$IP_ADDR:~/End-To-End-Generative-Dialogue/
scp -i ~/.ssh/gcloud-sshkey ~/installs/lua---parallel/init.lua michaelfarrell@$IP_ADDR:~/lua---parallel/

echo "bash startup.sh; cd lua---parallel; luarocks make; cd ../End-To-End-Generative-Dialogue; unzip data.zip; cd src; python preprocess.py" | ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey michaelfarrell@$IP_ADDR
