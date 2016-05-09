#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the data for the LSTM.
"""

import sys
import os

def child(ip_addr):
    os.system('scp -i ~/.ssh/gcloud-sshkey startup.sh michaelfarrell@%s:~/' % ip_addr)
    os.system('echo "bash startup.sh" | ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey michaelfarrell@%s' % ip_addr)
    os.system('scp -r -i ~/.ssh/gcloud-sshkey ../data.zip michaelfarrell@%s:~/End-To-End-Generative-Dialogue/' % ip_addr)
    os.system('scp -i ~/.ssh/gcloud-sshkey ../stash/parallel/init.lua michaelfarrell@%s:~/lua---parallel/' % ip_addr)
    os.system('echo "bash startup.sh; cd lua---parallel; luarocks make; cd ../End-To-End-Generative-Dialogue; unzip data.zip; cd src; python preprocess.py" | ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey michaelfarrell@%s' % ip_addr)
    os._exit(0)  


def main(arguments):
    with open('../client_list.txt') as f:
        for line in f:
            # os.system('echo ' + line)
            newpid = os.fork()
            if newpid == 0:
                if line[-1] ==  '\n':
                    child(line[:-1])
                else:
                    child(line)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
