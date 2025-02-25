#!/usr/bin/env python3

import sys
import pathlib
from os import listdir
from os.path import join, isdir
import json


cwd = str(pathlib.Path(__file__).parent.resolve())
config_file = "config.json"
prune_no_results = True

show_attributes = ["description"]
if len(sys.argv) > 1:
    show_attributes += [arg for arg in sys.argv[1:] if "description" not in arg]


subdirs = [subdir for subdir in listdir(cwd) if isdir(subdir)]
subdirs.sort()
for subdir in subdirs:
    cfile = join(subdir, config_file)
    try:
        with open(cfile, "r") as f:
            core_attributes = json.loads(f.read())
    except FileNotFoundError:
        continue
    if prune_no_results and len(listdir(join(subdir, "results"))) == 0:
        continue

    header = f"### {subdir:^50s} ###"
    to_print = f"{'#' * len(header)}\n{header}\n{'#' * len(header)}\n"
    for attr in show_attributes:
        if attr in core_attributes:
            val = core_attributes[attr]
        elif f"_{attr}" in core_attributes:
            val = core_attributes[f"_{attr}"]
        else:
            val = "Not Found"
        to_print += f"{attr:<50s}: {val}\n"

    print(to_print)
