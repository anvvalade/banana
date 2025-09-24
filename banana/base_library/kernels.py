from typing import Any, Callable, Optional

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from banana import BaseKernel, BaseVariable, f_cast, i_cast, jnp_ndarray

###########
# KERNELS #
###########


class KernelHMC(BaseKernel):
    """
    See BaseKernel and tensorflow_probability.mcmc.HamiltonianMonteCarlo
    """

    hmc_kwargs = {"num_leapfrog_steps": 512, "step_size": 0.01}
    trace_return = [
        "log_prob",
        "is_accepted",
        "step_size",
        "num_leapfrog_steps",
        # "imom_dot_fmom",
        "log_accept_ratio",
    ]

    def __init__(
        self,
        varinstances_sorted,
        log_prob_fn,
        verbosity,
        **kwargs,
    ):
        super().__init__(
            varinstances_sorted,
            log_prob_fn,
            verbosity=verbosity,
            **kwargs,
        )
        self._hmc_kwargs = self.hmc_kwargs.copy()

        self._hmc_kwargs["step_size"] = [
            mm / self._hmc_kwargs["num_leapfrog_steps"] for mm in self._mass_matrix
        ]
        self._logger.assertIsNotIn(
            None, "None", self._hmc_kwargs.values(), "hmc_kwargs"
        )

    def trace(self, chain_state, prev_kern_res):
        """ """
        # print(dir(prev_kern_res))
        # print(prev_kern_res)

        prev_kern_res = self._getInnermostResults(prev_kern_res)
        log_prob = prev_kern_res.accepted_results.target_log_prob
        is_accepted = i_cast(prev_kern_res.is_accepted)
        step_size = jnp.squeeze(
            prev_kern_res.accepted_results.step_size[0][0]
            / self._hmc_kwargs["step_size"][0][0]
        )
        num_leapfrog_steps = prev_kern_res.accepted_results.num_leapfrog_steps
        log_acc_rat = prev_kern_res.log_accept_ratio
        # imom = prev_kern_res.initial_momentum
        # fmom = prev_kern_res.final_momentum
        # imom_dot_fmom = tfm.reduce_sum(imom * fmom) / tfm.sqrt(
        #     tfm.reduce_sum(tfm.pow(imom, 2)) * tfm.reduce_sum(tfm.pow(fmom, 2))
        # )
        # imom_dot_fmom = 0.0
        return (log_prob, is_accepted, step_size, num_leapfrog_steps, log_acc_rat)

    def getKernel(self):
        return self.maybeTransformKernel(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self._log_prob_fn,
                store_parameters_in_results=True,
                **self._hmc_kwargs,
            )
        )


class KernelNUTS(BaseKernel):
    """
    See BaseKernel and tensorflow_probability.mcmc.NoUTurnSampler
    """

    nuts_kwargs = {
        "max_tree_depth": 10,
        "unrolled_leapfrog_steps": 1,
        "max_energy_diff": 1e3,
        "parallel_iterations": 1,
        "step_size": 1.0,
    }
    trace_return = [
        "log_prob",
        "is_accepted",
        "step_size",
        "leapfrogs_taken",
        "has_divergence",
        "reach_max_depth",
        "accept_ratio",
    ]

    def __init__(
        self,
        varinstances_sorted,
        log_prob_fn,
        verbosity,
        **kwargs,
    ):
        super().__init__(
            varinstances_sorted,
            log_prob_fn,
            verbosity=verbosity,
            **kwargs,
        )

        self._nuts_kwargs = self.nuts_kwargs.copy()

        ss_fac = self._nuts_kwargs["step_size"] / 2 ** (
            self._nuts_kwargs["max_tree_depth"] - 1
        )

        self._nuts_kwargs["step_size"] = [ss_fac * mm for mm in self._mass_matrix]

        self._logger.assertIsNotIn(
            None, "None", self._nuts_kwargs.values(), "nuts_kwargs"
        )

    def trace(self, chain_state, prev_kern_res):
        prev_kern_res = self._getInnermostResults(prev_kern_res)
        log_prob = prev_kern_res.target_log_prob
        is_accepted = i_cast(prev_kern_res.is_accepted)
        step_size = jnp.squeeze(
            prev_kern_res.step_size[0][0] / self._nuts_kwargs["step_size"][0][0]
        )
        leapfrogs_taken = i_cast(prev_kern_res.leapfrogs_taken)
        has_divergence = i_cast(prev_kern_res.has_divergence)
        reach_max_depth = i_cast(prev_kern_res.reach_max_depth)
        acc_rat = jnp.exp(prev_kern_res.log_accept_ratio)
        return (
            log_prob,
            is_accepted,
            step_size,
            leapfrogs_taken,
            has_divergence,
            reach_max_depth,
            acc_rat,
        )

    def getKernel(self):
        """ """
        return self.maybeTransformKernel(
            tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=self._log_prob_fn,
                **self._nuts_kwargs,
            )
        )


#####################################
# KERNELS WITH STEP SIZE ADAPTATION #
#####################################

###############################
# SIMPLE STEP SIZE ADAPTATION #
###############################


class KernelStepSizeAdaptHMC(KernelHMC):
    """
    See BaseKernel and tensorflow_probability.mcmc.SimpleStepSizeAdpatation
    """

    step_adapt_kwargs = {
        "num_adaptation_steps": None,
        "target_accept_prob": 0.65,
        "adaptation_rate": 0.01,
    }

    def __init__(
        self,
        varinstances_sorted,
        log_prob_fn,
        verbosity,
        **kwargs,
    ):
        super().__init__(
            varinstances_sorted,
            log_prob_fn,
            verbosity=verbosity,
            **kwargs,
        )
        self._step_adapt_kwargs = self.step_adapt_kwargs.copy()

        # self.dual_avg_kwargs["shrinkage_target"] = [
        #     f_cast(self.dual_avg_kwargs["shrinkage_target"]) * mm
        #     for mm in self._mass_matrix
        # ]
        self._logger.assertIsNotIn(
            None, "None", self._step_adapt_kwargs.values(), "step_adapt_kwargs"
        )

    def getKernel(self):
        """ """
        return tfp.mcmc.SimpleStepSizeAdaptation(
            super().getKernel(), **self._step_adapt_kwargs
        )


class KernelStepSizeAdaptNUTS(KernelNUTS):
    """
    See BaseKernel and tensorflow_probability.mcmc.SimpleStepSizeAdpatation
    """

    step_adapt_kwargs = {
        "num_adaptation_steps": None,
        "target_accept_prob": 0.5,
        "adaptation_rate": 0.01,
    }

    def __init__(
        self,
        varinstances_sorted,
        log_prob_fn,
        verbosity,
        **kwargs,
    ):
        super().__init__(
            varinstances_sorted,
            log_prob_fn,
            verbosity=verbosity,
            **kwargs,
        )
        self._step_adapt_kwargs = self.step_adapt_kwargs.copy()
        # self.dual_avg_kwargs["shrinkage_target"] = [
        #     f_cast(self.dual_avg_kwargs["shrinkage_target"]) * mm
        #     for mm in self._mass_matrix
        # ]

        self._logger.assertIsNotIn(
            None, "None", self._step_adapt_kwargs.values(), "step_adapt_kwargs"
        )

    def getKernel(self):
        """ """
        return tfp.mcmc.SimpleStepSizeAdaptation(
            super().getKernel(), **self._step_adapt_kwargs
        )


#######################################
# DUAL AVERAGING STEP SIZE ADAPTATION #
#######################################


class KernelDualAveragingHMC(KernelHMC):
    """
    See BaseKernel and tensorflow_probability.mcmc.DualAveragingStepSizeAdaptation
    """

    dual_avg_kwargs = {
        "num_adaptation_steps": None,
        "target_accept_prob": 0.65,
        "exploration_shrinkage": 0.05,
        "shrinkage_target": 1.0,
        "step_count_smoothing": 10,
        "decay_rate": 0.75,
    }

    def __init__(
        self,
        varinstances_sorted,
        log_prob_fn,
        verbosity,
        **kwargs,
    ):
        super().__init__(
            varinstances_sorted,
            log_prob_fn,
            verbosity=verbosity,
            **kwargs,
        )
        self._dual_avg_kwargs = self.dual_avg_kwargs.copy()

        self._dual_avg_kwargs["shrinkage_target"] = [
            f_cast(self._dual_avg_kwargs["shrinkage_target"]) * ss
            for ss in self._hmc_kwargs["step_size"]
        ]
        self._logger.assertIsNotIn(
            None, "None", self._dual_avg_kwargs.values(), "dual_avg_kwargs"
        )

    def getKernel(self):
        """ """
        return tfp.mcmc.DualAveragingStepSizeAdaptation(
            super().getKernel(), **self._dual_avg_kwargs
        )


class KernelDualAveragingNUTS(KernelNUTS):
    """
    See BaseKernel and tensorflow_probability.mcmc.DualAveragingStepSizeAdaptation
    """

    dual_avg_kwargs = {
        "num_adaptation_steps": None,
        "target_accept_prob": 0.9,
        "exploration_shrinkage": 0.05,
        "shrinkage_target": 10.0,
        "step_count_smoothing": 10,
        "decay_rate": 0.75,
    }

    def __init__(
        self,
        varinstances_sorted,
        log_prob_fn,
        verbosity,
        **kwargs,
    ):
        super().__init__(
            varinstances_sorted,
            log_prob_fn,
            verbosity=verbosity,
            **kwargs,
        )
        self._dual_avg_kwargs = self.dual_avg_kwargs.copy()

        self._dual_avg_kwargs["shrinkage_target"] = [
            f_cast(self._dual_avg_kwargs["shrinkage_target"]) * ss
            for ss in self._nuts_kwargs["step_size"]
        ]

        self._logger.assertIsNotIn(
            None, "None", self._dual_avg_kwargs.values(), "dual_avg_kwargs"
        )

    def getKernel(self):
        """ """
        return tfp.mcmc.DualAveragingStepSizeAdaptation(
            super().getKernel(),
            num_adaptation_steps=200,
            target_accept_prob=0.95,  # **self._dual_avg_kwargs
        )
