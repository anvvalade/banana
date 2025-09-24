#! /bin/env python

from importlib.machinery import SourceFileLoader
import inspect

from banana import ModelAgnosticChainsAnalyzer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("compute_Rhat_mass_matrix")
    parser.add_argument("root_dir")
    parser.add_argument("basename_sub_dir", help="The wildcard '#' will be resolved")
    parser.add_argument(
        "quantity", help="The name of the quantity (divv_modes, distances, etc)"
    )
    parser.add_argument("-c", "--chain", default="main")
    parser.add_argument(
        "--load", default="one_by_one", choices=["one_by_one", "all_at_once"]
    )
    parser.add_argument(
        "-r",
        "--randomize",
        type=int,
        default=100,
        help="Number of random trials to estimate statistical variation of the metrics (0 for none)",
    )
    parser.add_argument(
        "--show_top_k",
        type=int,
        default=10,
        help="Number of individual values to prompt when showing summary statistics",
    )
    parser.add_argument(
        "--save_mean_std",
        type=str,
        help="Basename of the file (mean / std and quantiy names are added)",
    )
    parser.add_argument(
        "-ff",
        "--functions_file",
        type=str,
        default=None,
        help="Python file of functions to run. Each function should take a state and return a dict(key1=val1, key2=val2, ...)",
    )

    args = parser.parse_args()

    if args.functions_file is not None:
        lib = SourceFileLoader(args.functions_file, args.functions_file).load_module()
        # Loading all the classes in the file
        functions = dict(inspect.getmembers(lib, inspect.isfunction)).values()
    else:
        functions = {}

    MACA = ModelAgnosticChainsAnalyzer(
        args.root_dir,
        args.basename_sub_dir,
        args.chain,
        args.quantity,
        args.load,
        0,  # args.randomize,
        functions,
    )

    MACA.summarize(args.show_top_k)
    MACA.saveMeanStd(args.save_mean_std)
