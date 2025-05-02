#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
config_topdir = Path("/home/tt544/tier_rout/configs")

with open(path, 'r') as fdesc:
    config = json.load(fdesc)

while path != config_topdir:
    path = path.parent.absolute()
    train_file = str(path) + "/train.json"
    if os.path.exists(train_file):
        with open(train_file, 'r') as fdesc:
            attrs = json.load(fdesc)
            for k,v in attrs.items():
                config[k] = v

for k,v in config.items():
    if isinstance(v, list):
        print(f"--{k} ", end = " ")
        for item in v:
            print(item, end = " ")
    else:
        print(f"--{k} {v}", end = " ")