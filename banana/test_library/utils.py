from banana import BaseUtil, ScalarType


from astropy.constants import c
from astropy.cosmology import LambdaCDM

c_light = c.value / 1000.0  # (km / s)


class RedshiftDistance(BaseUtil):

    def __init__(self, Omega_m: ScalarType, h0: ScalarType, verbosity=2):
        super().__init__(verbosity)

        self.h0 = h0
        self.LCDM = LambdaCDM(H0=h0 * 100, Om0=Omega_m, Ode0=1 - Omega_m)

    def getDistances(self, z_tot):
        return self.LCDM.comoving_distance(z_tot) * self.h0
