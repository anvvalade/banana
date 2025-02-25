from typing import Union
from banana import BaseModule, f_cast, DataType, ScalarType
import jax.numpy as jnp
from utils import c_light
import matplotlib.pyplot as plt


class CosmologicalRedshifts(BaseModule):
    provides = "z_cosmo"
    math_provides = r"z_{{\rm cos}, i}"

    def __init__(
        self,
        Omega_m: float,
        verbosity=2,
    ):
        super().__init__(
            verbosity,
        )
        self.factor_a = f_cast(2.0 / (3.0 * Omega_m))
        self.factor_b = f_cast(3 * Omega_m * 100.0 / c_light)

    def call(self, distances):
        return self.factor_a * (1 - jnp.sqrt(1 - self.factor_b * distances))


class Redshifts(BaseModule):
    provides = "z_tot"
    math_provides = r"z_i"

    def __init__(
        self,
        verbosity=2,
    ):
        super().__init__(
            verbosity,
        )
        self.inv_c_light = f_cast(1 / c_light)
        self._logger.debug(f"1 / c = {self.inv_c_light}")

    def call(self, z_cosmo, v_r):
        return (1 + z_cosmo) * (1 + v_r * self.inv_c_light) - 1

    def debug(self, z_cosmo, v_r):
        self._logger.descriptiveStatisticsTable(
            **{"1 + z_cosmo": 1 + z_cosmo, "1 + z_pec": 1 + v_r * self.inv_c_light}
        )
        fig, ax = plt.subplots()
        ax.scatter(z_cosmo, v_r, s=1)
        self._savePlot(fig, "debug", "z_cosmo_test")

        return (1 + z_cosmo) * (1 + v_r * self.inv_c_light) - 1

    def details(self, z_cosmo, v_r):
        z_cosmo_p1 = 1 + z_cosmo
        z_pec_p1 = 1 + v_r * self.inv_c_light

        return dict(
            z_tot=(1 + z_cosmo) * (1 + v_r * self.inv_c_light) - 1,
            z_cosmo_p1=z_cosmo_p1,
            z_pec_p1=z_pec_p1,
        )


class DistanceModuli(BaseModule):
    provides = "dist_mod"
    math_provides = r"\mu_i"

    def __init__(
        self,
        h0: float,
        verbosity=2,
    ):
        super().__init__(
            verbosity,
        )

        self.log_h0 = f_cast(5 * jnp.log10(h0))

    def call(self, z_cosmo, distances):
        return 5 * jnp.log10((1 + z_cosmo) * distances) + 25.0 + self.log_h0


class LikelihoodDistanceModuli(BaseModule):
    provides = "log_prob"
    math_provides = r"\log P(\mu^{\rm obs} | ...)"

    def __init__(
        self,
        dist_mod: DataType,
        err_dist_mod: Union[DataType, ScalarType],
        verbosity=2,
    ):
        super().__init__(
            verbosity,
        )
        self.dist_mod = f_cast(dist_mod)
        self.exp_inv_fac = f_cast(1 / (2 * err_dist_mod**2))

    def call(self, dist_mod):
        return -jnp.sum(((dist_mod - self.dist_mod) ** 2) * self.exp_inv_fac)


class LikelihoodRedshifts(BaseModule):
    provides = "log_prob"
    math_provides = r"\log P(\z_tot^{\rm obs} | ...)"

    def __init__(
        self,
        z_tot: DataType,
        err_z_tot: Union[DataType, ScalarType],
        verbosity=2,
    ):
        super().__init__(
            verbosity,
        )
        self.z_tot = f_cast(z_tot)
        self.exp_inv_fac = f_cast(1 / (2 * err_z_tot**2))

    def call(self, z_tot):
        return -jnp.sum(((z_tot - self.z_tot) ** 2) * self.exp_inv_fac)


class PriorDistances(BaseModule):
    provides = "log_prob"
    math_provides = r"\log P(d)"

    def call(self, distances):
        return jnp.sum(2 * jnp.log(distances))


class PriorVelocities(BaseModule):
    provides = "log_prob"
    math_provides = r"\log P(v)"

    def __init__(
        self,
        sigma_v: ScalarType,
        verbosity=2,
    ):
        super().__init__(
            verbosity,
        )
        self.exp_inv_fac = f_cast(1 / (2 * sigma_v**2))

    def call(self, v_r):
        return jnp.sum(-(v_r**2) * self.exp_inv_fac)
