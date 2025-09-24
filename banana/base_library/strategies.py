from banana.baseclasses import BaseStrategy
from banana.utils import getKeysFromDicts


class StrategyMaximizeLogProb(BaseStrategy):
    names = ["run_debug", "maximize", "analyze"]

    commands = {
        "run_debug": "runDebug",
        "maximize": "maximizeLogProb",
        "analyze": "runAnalysis",
    }

    cmd_kwargs = {
        "run_debug": {"run": True, "debug_classes": "none", "current_state": None},
        "maximize": {
            "run": True,
            "solver": None,
            "solver_kwargs": {},
            "variables_strategy": {},
            "keep_failed": True,
            "return_results": "best",
            "n_walkers": 1,
        },
        "analyze": {"name_run": "maximize"},
    }


class StrategyWarmupMain(BaseStrategy):
    """ """

    names = [
        "generate_initial_state",
        "run_debug",
        "run_kernel_warmup",
        "save_warmup",
        "summarize_warmup",
        "forget_warmup",
        "run_kernel_main",
        "save_main",
        "summarize_main",
    ]
    commands = {
        "generate_initial_state": "generateInitialState",
        "run_debug": "runDebug",
        "run_kernel_warmup": "runKernel",
        "save_warmup": "saveResults",
        "summarize_warmup": "summarize",
        "forget_warmup": "forgetChain",
        "run_kernel_main": "runKernel",
        "save_main": "saveResults",
        "summarize_main": "summarize",
    }
    sample_chain_kwargs = {
        "num_results": 0,
        "num_burnin_steps": 0,
        "num_steps_between_results": 1,
        "return_final_kernel_results": True,
        "seed": None,
    }
    cmd_kwargs = {
        "generate_initial_state": {"seed": None},
        "run_debug": {"run": True, "debug_classes": "none", "current_state": None},
        "run_kernel_warmup": {
            "name_run": "warmup",
            "jit_compile": True,
            "current_state": None,
            "last_kernel_results": None,
            "kernel": None,
            "sample_chain_kwargs": sample_chain_kwargs.copy(),
        },
        "save_warmup": {
            "command": "runKernel",
            "name_run": "warmup",
            "current_state": None,
            "last_kernel_results": None,
            "save_mode": "overwrite",
        },
        "summarize_warmup": {
            "name_run": "warmup",
        },
        "forget_warmup": {"name_run": "warmup"},
        "run_kernel_main": {
            "name_run": "main",
            "jit_compile": True,
            "current_state": None,
            "last_kernel_results": None,
            "kernel": None,
            "time_per_step": None,
            "sample_chain_kwargs": sample_chain_kwargs.copy(),
        },
        "save_main": {
            "command": "runKernel",
            "name_run": "main",
            "current_state": None,
            "last_kernel_results": None,
            "save_mode": "overwrite",
        },
        "summarize_main": {
            "name_run": "main",
        },
    }


class StrategyFullRun(BaseStrategy):
    names = [
        "generate_initial_state",
        "run_debug",
        "run_kernel_warmup",
        "save_warmup",
        "summarize_warmup",
        "analyze_trace_warmup",
        "forget_warmup",
        "run_kernel_main",
        "save_main",
        "summarize_main",
        "analyze_trace_main",
        "analyze_main",
    ]
    commands = {
        "generate_initial_state": "generateInitialState",
        "run_debug": "runDebug",
        "run_kernel_warmup": "runKernel",
        "save_warmup": "saveResults",
        "summarize_warmup": "summarize",
        "analyze_trace_warmup": "runTraceAnalysis",
        "forget_warmup": "forgetChain",
        "run_kernel_main": "runKernel",
        "save_main": "saveResults",
        "summarize_main": "summarize",
        "analyze_trace_main": "runTraceAnalysis",
        "analyze_main": "runAnalysis",
    }
    sample_chain_kwargs = {
        "num_results": 0,
        "num_burnin_steps": 0,
        "num_steps_between_results": 1,
        "return_final_kernel_results": True,
        "seed": None,
    }
    cmd_kwargs = {
        "generate_initial_state": {"seed": None},
        "run_debug": {"run": True, "debug_classes": "none", "current_state": None},
        "run_kernel_warmup": {
            "name_run": "warmup",
            "jit_compile": True,
            "current_state": None,
            "last_kernel_results": None,
            "kernel": None,
            "sample_chain_kwargs": sample_chain_kwargs.copy(),
        },
        "save_warmup": {
            "command": "runKernel",
            "name_run": "warmup",
            "current_state": None,
            "last_kernel_results": None,
            "save_mode": "overwrite",
        },
        "summarize_warmup": {
            "name_run": "warmup",
        },
        "analyze_trace_warmup": {"name_run": "warmup"},
        "forget_warmup": {"name_run": "warmup"},
        "run_kernel_main": {
            "name_run": "main",
            "jit_compile": True,
            "current_state": None,
            "last_kernel_results": None,
            "kernel": None,
            "time_per_step": None,
            "sample_chain_kwargs": sample_chain_kwargs.copy(),
        },
        "save_main": {
            "command": "runKernel",
            "name_run": "main",
            "current_state": None,
            "last_kernel_results": None,
            "save_mode": "overwrite",
        },
        "summarize_main": {
            "name_run": "main",
        },
        "analyze_trace_main": {"name_run": "main"},
        "analyze_main": {"name_run": "main"},
    }


class StrategyExpandChain(BaseStrategy):

    names = ["load", "run_kernel", "save", "summarize", "analyze"]
    commands = {
        "load": "loadForRerun",
        "run_kernel": "runKernel",
        "save": "saveResults",
        "summarize": "summarize",
        "analyze": "runAnalysis",
    }
    sample_chain_kwargs = {
        "num_results": 0,
        "num_burnin_steps": 0,
        "num_steps_between_results": 1,
        "return_final_kernel_results": True,
        "seed": None,
    }
    cmd_kwargs = {
        "load": {"name_run": None},
        "run_kernel": {
            "name_run": None,
            "jit_compile": True,
            "current_state": None,
            "last_kernel_results": None,
            "kernel": None,
            "time_per_step": None,
            "sample_chain_kwargs": sample_chain_kwargs.copy(),
        },
        "save": {
            "command": "runKernel",
            "name_run": None,
            "current_state": None,
            "last_kernel_results": None,
            "save_mode": "append",
        },
        "summarize": {
            "name_run": None,
        },
        "analyze": {"name_run": None},
    }


class StrategyLoadAnalyze(BaseStrategy):

    names = [
        "load",
        "summarize",
    ]
    commands = {
        "load": "loadResults",
        "summarize": "summarize",
    }
    cmd_kwargs = {
        "load": {"name_run": None, "kernel": None},
        "summarize": {"name_run": None},
    }


class StrategyLoadSummarize(BaseStrategy):

    names = [
        "load",
        "analyze",
    ]
    commands = {
        "load": "loadResults",
        "analyze": "runAnalysis",
    }
    cmd_kwargs = {
        "load": {"name_run": None, "kernel": None},
        "analyze": {"name_run": None, "kernel": None, "kernel_instance": None},
    }
