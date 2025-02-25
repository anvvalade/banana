#! /bin/env python

import time

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import jit
from tensorflow_probability.substrates import jax as tfp

from tqdm import tqdm, trange
import os

from .common import fdtype
from .utils import f_cast

exclude = [
    "has_divergence",
    "is_accepted",
    "leapfrogs_taken",
    "reach_max_depth",
    "step_size",
]


@jit
def stats(chain):
    mean = jnp.mean(chain, axis=0)
    std = jnp.std(chain, axis=0)
    ess = tfp.mcmc.effective_sample_size(chain)
    return mean, std, ess


# class Timer:

#     def __call__(self, message=None, restart=False):
#         if not hasattr(self, "t") or restart:
#             self.t = time.time()
#         else:
#             t = time.time()
#             if message is None:
#                 print(f"Timer: {t - self.t} since last call")
#             else:
#                 print(f"{message}: {t - self.t}s")
#             self.t = t


# timer = Timer()


class ModelAgnosticChainsAnalyzer:

    def __init__(
        self,
        root_dir,
        basename_sub_dir,
        chain_name,
        param_name,
        load_mode,
        randomize,
        target,
        functions,
    ):

        self.param_name = param_name
        # dirs_to_load = glob.glob(f"{root_dir}/{basename_sub_dir.replace('#', '*')}")
        # print(f"{root_dir}/{basename_sub_dir.replace('#', '*')}")

        self._findChains(root_dir, basename_sub_dir, chain_name, param_name)

        self._setTarget(target)
        self._setFunctions(functions)

        if load_mode == "one_by_one":
            for i, path in tqdm(
                enumerate(self.paths_to_chains),
                desc="Computing summary stats for each realization",
            ):
                self._loadOneChain(i, path)

            print("Combining results")
            self._combineResults()
        elif load_mode == "all_at_once":
            self._loadAllChains()

        if randomize > 0:
            self._computeRandomMetrics(randomize)
        else:
            self.Rhat_rand = None
            self.ess_rand = None

    def _findChains(self, root_dir, basename_sub_dir, chain_name, param_name):
        dirs_to_load = [
            dir
            for dir in os.listdir(root_dir)
            if (os.path.isdir(f"{root_dir}/{dir}") and basename_sub_dir in dir)
        ]

        chain_name = f"{chain_name}_" if chain_name != "" else ""
        self.paths_to_chains = [
            f"{root_dir}/{dir}/results/{chain_name}{param_name}.npy"
            for dir in dirs_to_load
        ]
        print(self.paths_to_chains)
        self.paths_to_chains = [p for p in self.paths_to_chains if os.path.isfile(p)]
        print(
            f"Trying to load {len(self.paths_to_chains)} files {load_mode}:\n- "
            + "\n- ".join(self.paths_to_chains)
        )

        if len(self.paths_to_chains) == 0:
            raise ValueError("Found no chain to load!")
        elif len(self.paths_to_chains) == 1:
            print("WARNING: only one chain, Rhat is not accurate!")

    def _setTarget(self, target):
        pass

    def _setFunctions(self, functions):
        self.functions = functions
        self.functions.update({"identity": lambda x: {self.param_name: x}})

        self.f_means, self.f_stds, self.f_esss = {}, {}, {}
        self.f_mean, self.f_std, self.f_mess, self.f_rhat = {}, {}, {}, {}
        self.f_size = {}

    @staticmethod
    def _reshapeChain(chain):
        if chain.ndim < 2:
            chain = np.reshape(chain, (chain.size, 1))
        elif chain.ndim > 2:
            chain = np.reshape(chain, (chain.shape[0], np.prod(chain.shape[1:])))

        return chain

    @classmethod
    def _load(cls, path):
        try:
            return cls._reshapeChain(f_cast(np.load(path, allow_pickle=False)))
        except FileNotFoundError:
            print(f"\nFile {path} does not exist\n")
            return

    def _updateOneChain(self, i, chain):
        if i == 0:
            self.n_chains = 0
            self.n_states = chain.shape[0]
            self.n_params = chain.shape[-1]
        else:
            self.n_chains += 1

        for super_fname, func in self.functions.items():
            try:
                f_chain_dict = func(chain)
            except Exception as e:
                print(f"Failed to run {super_fname}, got " + ",".join(e.args))
                continue

            for fname, f_chain in f_chain_dict.items():
                mean, std, ess = stats(f_chain)

                if i == 0:
                    self.f_means[fname] = jnp.stack([mean])
                    self.f_stds[fname] = jnp.stack([std])
                    self.f_esss[fname] = jnp.stack([ess])

                    self.f_size[fname] = mean.size
                else:
                    self.f_means[fname] = jnp.concatenate(
                        [self.f_means[fname], mean[None, :]], axis=0
                    )
                    self.f_stds[fname] = jnp.concatenate(
                        [self.f_stds[fname], std[None, :]], axis=0
                    )
                    self.f_esss[fname] = jnp.concatenate(
                        [self.f_esss[fname], ess[None, :]], axis=0
                    )

    def _loadOneChain(self, i, path):

        if (chain := self._load(path)) is None:
            return

        self._updateOneChain(i, chain)

    def _combineResults(self):

        for fname in self.f_means:
            means = self.f_means[fname]
            stds = self.f_stds[fname]
            esss = self.f_esss[fname]

            if len(means) > 1:
                mean = jnp.mean(means, axis=0)
                std = jnp.sqrt(jnp.mean(stds**2 + means**2, axis=0) - mean**2)
                mean_ess = jnp.mean(esss, axis=0)
                rhat = jnp.mean(stds, axis=0) / std
            else:
                mean = means[0]
                std = stds[0]
                mean_ess = esss[0]
                rhat = 1.0

            self.f_mean[fname] = mean
            self.f_std[fname] = std
            self.f_mess[fname] = mean_ess
            self.f_rhat[fname] = rhat

        # Freeing memory
        del self.f_means
        del self.f_stds
        del self.f_esss

    def _loadAllChains(self):
        chains = []
        for path in tqdm(self.paths_to_chains, desc="Loading all realizations"):
            if (chain := self._load(path)) is not None:
                chains.append(chain)

        concat_chains = jnp.concatenate(chains, axis=0)
        self._updateOneChain(0, concat_chains)

        # we overwrite n_chains
        self.n_chains = len(chains)

        self._combineResults()

    def _computeRandomMetrics(self, n_rand):
        rhats = np.zeros(n_rand)
        esss = np.zeros(n_rand)
        for i in trange(n_rand, desc="Computing random metrics"):
            rand_chains = [
                np.random.normal(loc=0, size=self.n_states)
                for _ in range(self.n_chains)
            ]
            rhats[i] = R_hat(
                rand_chains,
                m=np,
                axis=None,
            )
            esss[i] = np.mean(
                np.stack(
                    [tfp.mcmc.effective_sample_size(chain) for chain in rand_chains]
                ),
            )

        self.Rhat_rand = np.quantile(rhats, [0.025, 0.975])
        self.ess_rand = np.quantile(esss, [0.025, 0.975])

    def _summarize(self, quant, quant_name, top_k, target=None, rand_interval=None):

        to_print = "\n" + "-" * len(quant_name) + "\n"
        to_print += f"{quant_name}\n"
        to_print += "-" * len(quant_name) + "\n"

        if target is not None:
            to_print += f"target       = {target:.5f}\n"
        if rand_interval is not None:
            to_print += (
                f"95% stat dev = {rand_interval[0]:.5f} - {rand_interval[-1]:.5f}\n\n"
            )

        if quant.size == 1:
            to_print += f"value        = {quant[0]:.5f}\n"
        else:
            mean = jnp.mean(quant)
            min, max = jnp.min(quant), jnp.max(quant)
            ql, med, qh = jnp.quantile(quant, f_cast([0.025, 0.5, 0.975]))

            to_print += f"mean         = {mean:.5f}\n"
            to_print += f"median       = {med:.5f}\n"
            to_print += f"min          = {min:.5f}\n"
            to_print += f"max          = {max:.5f}\n"
            to_print += f"95% interval = {ql:.5f} - {qh:.5f}\n"

            if top_k > 0:
                asort = jnp.argsort(quant)
                if top_k > self.n_params:
                    top_k = self.n_params

                best, worst = asort[-top_k:][::-1], asort[:top_k]
                to_print += f"\ntop {top_k} values:\n"
                for ind in best:
                    to_print += f"    {ind:<10d} {quant[ind]:.5f}\n"

                to_print += f"\nbottom {top_k} values:\n"
                for ind in worst:
                    to_print += f"    {ind:<10d} {quant[ind]:.5f}\n"

        print(to_print)

    def summarize(self, top_k):
        for fname in self.f_mean:
            to_print = f"### Summary of {fname} ###"
            to_print = (
                "#" * len(to_print) + f"\n{to_print}\n" + "#" * len(to_print) + "\n"
            )

            print(to_print)
            self._summarize(self.f_mean[fname], "Mean", top_k)
            self._summarize(self.f_std[fname], "Standard deviation", top_k)
            self._summarize(
                self.f_mess[fname],
                "Effective Sample Size",
                top_k,
                self.n_states,
                self.ess_rand,
            )
            self._summarize(self.f_rhat[fname], "R hat", top_k, 1.0, self.Rhat_rand)

    @staticmethod
    def _maybeMakeDirs(path):
        dir = path[: path.rfind("/")]
        os.makedirs(dir, exist_ok=True)

    def saveMeanStd(self, path):

        if path is None:
            return

        assert "#" in path, "Missing '#' placeholder for functions' names in path"
        assert path[-4:] == ".npy", "path is not an npy file"
        self._maybeMakeDirs(path)

        for what in ["mean", "std"]:
            data = getattr(self, f"f_{what}")
            for fname in self.f_mean:
                file = path.replace("#", f"{what}_{fname}")
                np.save(file, data[fname])
