from typing import Union
from banana import BaseVariable, DataType, ScalarType, f_cast
from utils import c_light
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


from utils import RedshiftDistance


class Distances(BaseVariable):
    provides = "distances"
    math_provides = "d_i"

    def __init__(
        self,
        z_tot: DataType,
        sigma_v: ScalarType,
        err_dist_mod: Union[DataType, ScalarType],
        RD: RedshiftDistance,
        verbosity=2,
    ):
        super().__init__(
            verbosity,
        )

        self.output_size = z_tot.size

        d_z = f_cast(RD.getDistances(z_tot))
        err_d = f_cast(
            jnp.minimum(sigma_v / 100.0, d_z * err_dist_mod * jnp.log(10) / 5)
        )

        self.setPriorMeanStdAndMM(d_z, err_d)
        self.init_state_gen = tfd.Normal(
            loc=self.prior_mean,
            scale=self.prior_std,
        )

    def summarize(self, chain, name_run):
        super().summarize(
            chain,
            name_run,
            plot_vs=self.prior_mean,  # type: ignore
            plot_vs_label="$d_z$ Mpc/h",
            plot_vs_func="plot",
        )


class Velocities(BaseVariable):
    provides = "v_r"
    math_provides = "v_{r, i}"

    def __init__(
        self,
        sigma_v: ScalarType,
        DI: Distances,
        verbosity=2,
    ):
        super().__init__(
            verbosity,
        )

        self.output_size = DI.output_size

        self.d_z = DI.prior_mean

        self.setPriorMeanStdAndMM(
            jnp.zeros(self.output_size), sigma_v * jnp.ones(self.output_size)
        )
        self.init_state_gen = tfd.Normal(
            loc=self.prior_mean,
            scale=self.prior_std,
        )

    def summarize(self, chain, name_run):
        super().summarize(
            chain,
            name_run,
            plot_vs=self.d_z,  # type: ignore
            plot_vs_label="$d_z$ Mpc/h",
            plot_vs_func="plot",
        )
