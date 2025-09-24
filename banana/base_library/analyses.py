from typing import Dict, Any, List, Union
from banana import BaseAnalysis, printTable, toList, Rsplit, isInstance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from corner import corner


from tensorflow_probability.substrates import jax as tfp

from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d


class GeneralAnalysis(BaseAnalysis):

    quantities = []
    targets = dict()
    uncertainties = dict()
    plot_vs = dict()

    save = True

    show_quantiles = [0.025, 0.5 - 0.341, 0.5, 0.5 + 0.341, 0.975]
    max_show_individual = 8

    _figsize = (8, 8)
    exts = ["pdf"]

    def __init__(
        self,
        parameters: Dict[str, Any],
        data: Dict[str, Any],
        verbosity: int = 2,
    ) -> None:
        super().__init__(verbosity)

        self.parameters = parameters
        self.data = data

        # dict are not ordered, better use lists
        self.__class__.finalize_args = self.quantities  # type: ignore

        # if we are given lists, we make dicts out of them
        self.targets = (
            self.targets
            if isinstance(self.targets, dict)
            else {q: t for q, t in zip(self.quantities, self.targets)}
        )
        self.uncertainties = (
            self.uncertainties
            if isinstance(self.uncertainties, dict)
            else {q: t for q, t in zip(self.quantities, self.uncertainties)}
        )
        self.plot_vs = (
            self.plot_vs
            if isinstance(self.plot_vs, dict)
            else {q: t for q, t in zip(self.quantities, self.plot_vs)}
        )

        self._save = (
            self.save
            if isinstance(self.save, dict)
            else {
                q: (self.save in (True, False) or q in self.save)
                for q in self.quantities
            }
        )

        self._targets = {}
        self._uncertainties = {}
        self._plot_vs = {}

        for name in ["targets", "uncertainties", "plot_vs"]:
            spec = getattr(self, name)
            loc = getattr(self, "_" + name)

            spec.update({q: None for q in self.quantities if q not in spec})
            for quant, vquant in spec.items():
                if vquant is None:
                    loc[quant] = None
                    continue

                if isinstance(vquant, str):
                    loc[quant] = self.getKeysFromDicts(
                        vquant,
                        # parameters=parameters,
                        data=data,
                    )
                else:
                    loc[quant] = np.squeeze(vquant)  # type: ignore

                if name == "plot_vs":
                    values = toList(loc[quant])
                    loc[quant] = {q: np.atleast_1d(vq) for q, vq in zip(vquant, values)}
                else:
                    loc[quant] = np.atleast_1d(loc[quant])  # type: ignore

            self._logger.debug(f"{name} are {loc}")

    def _plotHist(
        self,
        quant,
        val,
        target=None,
        uncertainty=None,
        plot_vs=None,
        collapse_on="data",
        show_kdes=False,
    ):

        quantiles = (
            [self.show_quantiles[0] / 2]
            + self.show_quantiles
            + [1 - (1 - self.show_quantiles[-1]) / 2]
        )
        qs = np.quantile(val, quantiles)

        if qs[0] * qs[-1] < 0:
            qmax = np.max(np.abs([qs[0], qs[-1]]))
            _min, _max = -qmax, qmax
            sym = True
        else:
            _min, _max = qs[0], qs[-1]
            sym = False
        x = np.linspace(_min, _max, 100)

        fig, ax = plt.subplots(figsize=self._figsize)
        ax.axhline(0, color="k", lw=0.5)

        if sym:
            ax.axvline(0, color="k", lw=0.5)

        if target is not None and target.size == 1:
            ax.axvline(target, color="g", lw=0.5, label=f"target = {float(target):.2f}")
        elif target is not None:
            kde = gaussian_kde(target)(x)
            ax.plot(x, kde, "g", label="target")

        if uncertainty is not None:
            self._logger.warn("Uncertainty is ignored: not coded yet")

        if val.shape[-1] > 1:
            _val = val if collapse_on == "realizations" else val.T
            kdes = [gaussian_kde(v)(x) for v in _val]
        else:
            kdes = [gaussian_kde(np.squeeze(val))(x)]

        if show_kdes:
            if plot_vs is not None and collapse_on == "data":
                colors = plb.get_cmap("turbo")(np.linspace(0.1, 0.9, len(kdes)))
                colors = colors[np.argsort(plot_vs)]
            else:
                colors = ["k"] * len(kdes)

            for col, kde in zip(colors, kdes):
                ax.plot(x, kde, c=col, ls="-", lw=0.5, alpha=1 / np.sqrt(len(kdes)))

        q_kdes = np.quantile(kdes, [0.25, 0.5, 0.75], axis=0)
        m_kdes = np.mean(kdes, axis=0)

        ax.plot(x, q_kdes[2], c="k", ls=":", label="75%")
        ax.plot(x, q_kdes[1], c="k", ls="--", label="50%")
        ax.plot(x, q_kdes[0], c="k", ls=":", label="25%")
        ax.plot(x, m_kdes, c="r", ls="-", label="mean")

        for i in range(len(qs - 2) // 2 - 1):
            ql, qh = qs[1 + i], qs[-2 - i]
            beg, end = np.searchsorted(x, [ql, qh])
            lab = f"{self.show_quantiles[i]:.0%} - {self.show_quantiles[-1-i]:.0%}: {qh - ql:.5f}"
            ax.fill_between(
                x[beg:end], 0.0, m_kdes[beg:end], color="b", alpha=0.2, label=lab
            )

        m, s = np.mean(val), np.std(val)
        if sym:
            label = f"m, s, m/s = {m:.5f}, {s:.5f}, {m/s:.5f}"
        else:
            label = f"m, s = {m:.5f}, {s:.5f}"

        ax.axvline(m, color="g", ls="-.", label=label)

        fig.legend(loc="upper center", ncols=2)
        fig.suptitle(f"Collapsed on {collapse_on}")
        ax.set_xlabel(quant)
        ax.set_ylabel(f"P({quant})")
        self._savePlot(fig, "anal", f"hist_{collapse_on}_{quant}", exts=self.exts)

    def _describe(self, x, y, nbins):
        ql, qh = np.quantile(x, [0.01, 0.99])
        # We create two extra bins on the side
        bins = np.concatenate([[-np.inf], np.linspace(ql, qh, nbins + 1), [np.inf]])
        # dig in [1, nbins + 2]
        # 1 -> left most bin, we dont care
        # n + 2 -> right most bin, we dont care
        dig = np.digitize(x, bins, right=False)
        values = np.zeros((nbins, len(self.show_quantiles)))
        for i in range(nbins):
            values[i, :] = np.quantile(y[dig == i + 2], self.show_quantiles)

        return 0.5 * (bins[2:-1] + bins[1:-2]), values.T

    def _plotVs(self, quant, val, quant_vs, val_vs):
        fig, ax = plt.subplots(figsize=self._figsize)

        if val_vs is None:
            self._logger.warn(
                f"Cannot plot {quant} vs {quant_vs}, the latter is 'None'"
            )
            return
        val_vs = np.stack([val_vs] * val.shape[0])

        qs = np.quantile(val, [0.01, 0.99])
        qvs = np.quantile(val_vs, [0.01, 0.99])

        if qs[0] * qs[-1] < 0:
            ax.axhline(0, ls="-", lw=0.5, color="k")

        if qvs[0] * qvs[-1] < 0:
            ax.axvline(0, ls="-", lw=0.5, color="k")

        mqs, dqs = 0.5 * (qs[0] + qs[1]), qs[1] - qs[0]
        mqvs, dqvs = 0.5 * (qvs[0] + qvs[1]), qvs[1] - qvs[0]
        if (
            dqs / dqvs > 0.9
            and dqs / dqvs < 1.1
            and np.abs(mqs - mqvs) < 0.5 * (dqs + dqvs) / 5
        ):
            # it's the same
            ax.set_aspect(1)
            qs[0] = qvs[0] = min([qs[0], qvs[0]])
            qs[1] = qvs[1] = max([qs[1], qvs[1]])
            ax.plot([qs[0], qs[1]], [qs[0], qs[1]], c="k", ls="-", lw=0.5)

        x = np.linspace(*qvs, 20)  # type: ignore
        y = np.linspace(*qs, 20)  # type: ignore

        h2d = np.histogram2d(val_vs.ravel(), val.ravel(), [x, y])[0]
        h2d = h2d / np.max(h2d)

        _x = 0.5 * (x[1:] + x[:-1])
        _y = 0.5 * (y[1:] + y[:-1])
        ax.contourf(
            _x, _y, h2d.T, levels=np.linspace(0.1, 1, 10), cmap="Greys", alpha=0.5
        )

        int_h2d = RegularGridInterpolator(
            (_x, _y), h2d, bounds_error=False, fill_value=0
        )
        w = int_h2d((val_vs, val)) < 0.1
        ax.scatter(val_vs[w], val[w], c="k", s=1, edgecolor="none", marker=".", alpha=1)

        x, qs = self._describe(val_vs, val, 10)

        for i in range(len(self.show_quantiles) // 2):
            ax.plot(x, qs[i], ls="--", color="b")
            ax.plot(x, qs[-1 - i], ls="--", color="b")

        ax.plot(x, qs[i + 1], ls="-", color="r")

        ax.set_xlabel(quant_vs)
        ax.set_ylabel(quant)

        self._savePlot(fig, "anal", f"{quant}_vs_{quant_vs}")

    def _plot(self, quant, val):  # , target=None, uncertainty=None, plot_vs=None):
        self._logger.debug(f"Plotting {quant} with shape {val.shape}")
        target = self._targets[quant]
        uncertainty = self._uncertainties[quant]
        plot_vs = self._plot_vs[quant]

        if val.shape[-1] == 1:
            self._plotHist(
                quant,
                val,
                target=target,
                uncertainty=uncertainty,
                plot_vs=plot_vs,
                collapse_on="realizations",
            )
        else:
            self._plotHist(
                quant,
                val,
                target=target,
                uncertainty=uncertainty,
                plot_vs=plot_vs,
                collapse_on="data",
            )
            if plot_vs is not None:
                for quant_vs, val_vs in plot_vs.items():
                    self._plotVs(quant, val, quant_vs, val_vs)

            if target is not None:
                diff = val - target[None, :]
                self._plotHist(
                    f"{quant}_to_target", diff, plot_vs=plot_vs, collapse_on="data"
                )
            if target is not None and plot_vs is not None:
                for quant_vs, val_vs in plot_vs.items():
                    self._plotVs(f"{quant}_to_target", diff, quant_vs, val_vs)

    def _plotCorner(self, **kwargs):
        quants = list(kwargs)
        vals = np.concatenate([kwargs[q] for q in quants], axis=-1)
        targets = np.array([float(self._targets[q]) for q in quants])

        cc = np.corrcoef(vals.T)
        self._logger.info(
            "Correlation matrix\n" + printTable(quants, quants, cc, data_fmt="%8.2f")
        )

        fig = corner(
            vals,
            labels=quants,
            show_titles=True,
            truths=targets,
            bins=int(np.sqrt(vals.shape[0])),
            hist_bin_factor=2,
            smooth=1,
            smooth1d=1,
        )
        self._savePlot(fig, "anal", f"corner_{'_'.join(quants)}")

    def _plotScalars(self, **kwargs):
        quants = list(kwargs)
        vals = [kwargs[q] for q in quants]
        targets = [self._targets[q] for q in quants]

        sq_n_quants = np.sqrt(len(quants))
        nrows = int(np.round(sq_n_quants))
        ncols = len(quants) // nrows + int((len(quants) % nrows) != 0)

        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(4 * ncols, 3 * nrows), sharex=True
        )

        axs = np.atleast_1d(axs)

        fig_corr, ax_corr = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        ax_corr.axhline(0, color="k")

        x = np.arange(1, vals[0].size + 1)
        for quant, val, tgt, ax in zip(quants, vals, targets, axs.flat):
            rsplit, mean, std = Rsplit(val)
            ax.plot(x, val, c="b")
            ax.axhline(mean, color="r", linestyle=":")
            if tgt is not None:
                ax.axhline(tgt, color="k", linestyle="--")

            labels = [
                f"{quant}",
                f"{float(mean):.4f} +- {float(std):.4f}",
                f"{float(tfp.mcmc.effective_sample_size(np.squeeze(val))):.2f} | {float(rsplit):.4f}",
                f"{float(tgt):.4f}" if tgt is not None else None,
            ]
            for lab, yy, xx in zip(
                labels, ["top", "bottom"] * 2, ["left"] * 2 + ["right"] * 2
            ):
                if lab is None:
                    pass
                ax.text(
                    0.05 if xx == "left" else 0.95,
                    0.05 if yy == "bottom" else 0.95,
                    lab,
                    ha=xx,
                    va=yy,
                    transform=ax.transAxes,
                )

            sval = np.squeeze(val) - np.mean(val)
            corr = gaussian_filter1d(
                [
                    np.sum(sval[: sval.size - n] * sval[n:])
                    for n in range(sval.size - 1)
                ],
                1,
            )
            corr /= corr[0]
            w0 = np.where(corr < 0)[0][0]
            ax_corr.plot(corr, label=f"{quant} " + r"$(\lambda_{0} = %d)$" % (w0))

        ax_corr.legend()
        ax_corr.set_xlabel("lag")
        ax_corr.set_ylabel(r"$\rho$")

        self._savePlot(fig, "anal", f"plot_{'_'.join(quants)}")
        self._savePlot(fig_corr, "anal", f"autocorr_{'_'.join(quants)}")

    def finalize(self, **kwargs):
        scalars = {}
        for quant, val in kwargs.items():
            if val.shape[-1] == 1:
                scalars[quant] = val

            self._plot(quant, val)
            if self._save[quant]:
                self._saveData(val, "res", f"main_{quant}", exts=["npy"])

        self._plotCorner(**scalars)
        self._plotScalars(**scalars)
