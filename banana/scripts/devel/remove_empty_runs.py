#!/usr/bin/env python3

import sys
import pathlib
from os import listdir
from os.path import join, isdir, isfile
from shutil import rmtree

cwd = str(pathlib.Path(__file__).parent.resolve())
config_file = "config.json"
prune_no_results = True

force = False
criterion = 'no_results'
if len(sys.argv) > 1:
    if '-f' in sys.args or '--force' in sys.args:
        force = True
    if any('--no_' in arg for arg in sys.args):
        criterion = 'no_'
        aux_criterion = [arg for arg in sys.args if '--no_' in arg][0]

subdirs = [subdir for subdir in listdir(cwd) if isdir(subdir)]
subdirs.sort()
to_remove = []
for subdir in subdirs:
    resdir = join(subdir, 'results')
    resfiles = [file for file in listdir(resdir) if isfile(file)]
    if criterion == 'no_results' and resfiles == []:
        to_remove.append(subdir)
    elif criterion == 'no_':
        selfiles = [file for file in resfiles if aux_criterion in file]
        if selfiles == []:
            to_remove.append(subdir)

print(f"These subdirs will be removed \n{'\n'.join(subdirs)}")

if not force:
    ans = stubbornInput("OK? (y, n) > ", ['y', 'n'])
    if ans == 'n':
        raise 

for subdir in to_remove:
    print(f"Removing {subdir}...")
    rmtree(subdir)
        
