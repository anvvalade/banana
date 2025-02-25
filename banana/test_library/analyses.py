from banana import BaseAnalysis
import matplotlib.pyplot as plt


class TestAnalysis(BaseAnalysis):

    def __init__(self, sigma_v: float, verbosity=2):
        super().__init__(verbosity=verbosity)
        self.sigma_v = sigma_v

    def _plot(self, real, v_r, z_pec_p1):
        fig, ax = plt.subplots()
        ax.scatter(v_r, z_pec_p1)
        self._savePlot(fig, "anal", f"{real}_v_r_z_pec_p1")

    def _save(self, real, z_pec_p1):
        self._saveData(z_pec_p1, "res", f"{real}_z_pec_p1", exts=["npy", "bin"])

    def oneState(self, real, v_r, z_pec_p1):
        if self.plot_level > 1:  # type: ignore
            self._plot(real, v_r, z_pec_p1)
        if self.save_level > 1:  # type: ignore
            self._save(real, z_pec_p1)

    def finalize(self, z_cosmo_p1):
        if self.plot_level > 0:  # type: ignore
            self._plot("last", z_cosmo_p1, z_cosmo_p1)  # whatever
        if self.save_level > 0:  # type: ignore
            # save some stuff once
            pass
