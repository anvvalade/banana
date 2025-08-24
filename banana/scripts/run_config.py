import os

os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import sys
from banana import BananaCore


import pathlib

current_path = str(pathlib.Path(__file__).parent.resolve())

if len(sys.argv) > 1:
    cfile = sys.argv[1]
else:
    raise ValueError(
        "This script takes at least one argument: the configuration file. "
    )

if len(sys.argv) > 2:
    argv = sys.argv[2:]
else:
    argv = []


BC = BananaCore(cfile, argv, verbosity=3)
BC.buildLogProb()
BC.runStrategy()
