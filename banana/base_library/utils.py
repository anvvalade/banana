import jax.numpy as jnp

import numpy as np
from typing import Optional, Tuple, Union

from banana.common import cdtype, jnp_ndarray, Jnp_ndarray
from banana.logger import Logger
from banana.baseclasses import BaseUtil


class FFT3DWrapper(BaseUtil):
    """ """

    def __init__(
        self,
        n_grid: int,
        L_grid: Union[float, int],
        k_min: Optional[float] = None,
        k_max: Optional[float] = None,
        verbosity: int = 2,
    ):
        super().__init__(verbosity)
        self.n_grid = n_grid
        self.L_grid = L_grid

        self._logger.debug(f"Generating modes from grid n / L = {n_grid} / {L_grid}")

        # Building k-grid
        kz = 2 * np.pi * np.fft.rfftfreq(n_grid, d=L_grid / n_grid)
        kxy = 2 * np.pi * np.fft.fftfreq(n_grid, d=L_grid / n_grid)
        self.kx_grid, self.ky_grid, self.kz_grid = np.meshgrid(
            kxy, kxy, kz, indexing="ij"
        )
        self.k_norm_grid = np.sqrt(self.kx_grid**2 + self.ky_grid**2 + self.kz_grid**2)
        self.k_grid_shape = self.k_norm_grid.shape
        self.n_modes_grid = self.k_norm_grid.size

        self._logger.debug(
            f"Number of modes of the grid {self.n_modes_grid} and spans on [0, {np.max(self.k_norm_grid)}]"
        )

        # Selecting modes
        # /!\ k = 0 is not selected by default!
        self.k_min = (
            np.min(self.k_norm_grid[self.k_norm_grid > 0]) if k_min is None else k_min
        )
        self.k_max = np.max(self.k_norm_grid) if k_max is None else k_max
        k_select = np.where(
            (self.k_norm_grid >= self.k_min) * (self.k_norm_grid <= self.k_max)
        )
        k_norm = self.k_norm_grid[k_select]
        k_asort = np.argsort(k_norm)
        self.k_select = tuple([ks[k_asort] for ks in k_select])

        self.kx = self.kx_grid[self.k_select]
        self.ky = self.ky_grid[self.k_select]
        self.kz = self.kz_grid[self.k_select]
        self.k_norm = self.k_norm_grid[self.k_select]
        self.n_modes = self.k_norm.size

        self._logger.debug(
            f"Number of modes selected within [{self.k_norm[0]}, {self.k_norm[-1]}] = {self.n_modes}"
        )

        self._logger.success("Instanciation ran smoothly")

    # From / to grid

    def gridToselected(self, grid: Jnp_ndarray) -> Tuple[jnp_ndarray]:
        """

        Parameters
        ----------
        grid: Jnp_ndarray :


        Returns
        -------


        """
        re = jnp.real(grid)
        im = jnp.imag(grid)
        return re[self.k_select], im[self.k_select]

    # From / to selected modes

    def selectedToGrid(self, re: Jnp_ndarray, im: Jnp_ndarray) -> Jnp_ndarray:
        """

        Parameters
        ----------
        re: Jnp_ndarray :

        im: Jnp_ndarray :


        Returns
        -------


        """
        gmodes = jnp.zeros(
            (self.n_grid, self.n_grid, self.n_grid // 2 + 1), dtype=cdtype
        )
        return gmodes.at[self.k_select].add(re + 1j * im)

    def irfftFromSelected(self, re: Jnp_ndarray, im: Jnp_ndarray) -> Jnp_ndarray:
        """

        Parameters
        ----------
        re: Jnp_ndarray :

        im: Jnp_ndarray :


        Returns
        -------


        """
        val = jnp.fft.irfftn(self.selectedToGrid(re, im), norm="ortho")
        return val

    def rfftToSelected(self, field: Jnp_ndarray) -> Jnp_ndarray:
        """

        Parameters
        ----------
        field: Jnp_ndarray :


        Returns
        -------


        """
        gmodes = jnp.fft.rfftn(field, norm="ortho")
        smodes = gmodes[self.k_select]
        return jnp.real(smodes), jnp.imag(smodes)

    # From / to concatenated modes

    def concatToGrid(self, reim: Jnp_ndarray) -> Jnp_ndarray:
        """

        Parameters
        ----------
        reim: Jnp_ndarray :


        Returns
        -------


        """
        return self.selectedToGrid(reim[: self.n_modes], reim[self.n_modes :])

    def irfftFromConcat(self, reim: Jnp_ndarray) -> Jnp_ndarray:
        """

        Parameters
        ----------
        reim: Jnp_ndarray :


        Returns
        -------


        """
        return jnp.fft.irfftn(self.concatToGrid(reim), norm="ortho")

    def rfftToConcat(self, field: Jnp_ndarray) -> Jnp_ndarray:
        """

        Parameters
        ----------
        field: Jnp_ndarray :


        Returns
        -------


        """
        return jnp.concatenate(self.rfftToSelected(field))
