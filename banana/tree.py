import networkx as nx
from .logger import Logger
import matplotlib.pyplot as plt
from matplotlib import colors as mpcl
import numpy as np
from tqdm import trange
from random import sample


class Tree:
    graph_radius = 20

    def __init__(
        self,
        layers,
        variables_layer,
        links,
        getClasses,
        io,
        verbosity: int = 2,
        **kwargs,
    ):
        self._logger = Logger("Tree", verbosity)
        self._layers = layers
        self._variables_layer = variables_layer
        self._links = links
        self._getClasses = getClasses
        self._io = io
        self.__dict__.update(kwargs)

    def print(self, embedded=False):
        """

        Parameters
        ----------
        embedded :
             (Default value = False)

        Returns
        -------


        """
        string = "\nNODES:\n"
        string += f"{'variables':<15} | " + " | ".join(self._variables_layer) + "\n"
        for i, layer in enumerate(self._layers[::-1]):
            left = f"Layer {len(self._layers) - 1 - i}"
            string += f"{left:<15} | " + " | ".join(layer) + "\n"

        string += "\nLINKS:\n"
        for client, all_providers in self._links.items():
            for provider in all_providers:
                string += f"{client:<60} <- {provider:<40}\n"

        if not embedded:
            print(string)
        else:
            return string
        # self._tsize = (get_terminal_size().rows,)
        # self._string2d = String2D(maxwidth=get_terminal_size().cols)
        # self._addLayers()
        # self._addParameters()
        # self._addModModLinks()
        # self._addModVarLinks()

    def _getName(self, name, mode):
        if mode == "normal":
            return name
        elif "short" in mode:
            shift = int(mode.replace("short_", ""))
            mask = [not ch.islower() for ch in name]
            return "".join(
                [
                    ch
                    for i, ch in enumerate(name)
                    if mask[i] or mask[max([0, i - shift])]
                ]
            )
        elif mode == "math":
            _cls = self._getClasses(name)
            if _cls.math_provides is not None:
                return f"${_cls.math_provides}$"
            else:
                return name
        elif mode == "debug":
            self._logger.Assert(
                hasattr(self, "_debug_info"),
                AttributeError(
                    "Cannot build debug graph without debug info. Please run core.runDebugLogProb() once before."
                ),
            )
        else:
            self._logger.error(
                ValueError(f"Mode {mode} not in full, short_[i], math or debug")
            )

    def _getFormatedLinks(self, mode):
        pairs = []
        self.formated_nodes = {}
        for cl, prs in self._links.items():
            # we store to avoid unnecessary recomputation
            if cl not in self.formated_nodes:
                cl_name = self.formated_nodes[cl] = self._getName(cl, mode)
            else:
                cl_name = self.formated_nodes[cl]
            for pr in prs:
                # we store to avoid unnecessary recomputation
                if pr not in self.formated_nodes:
                    pr_name = self.formated_nodes[pr] = self._getName(pr, mode)
                else:
                    pr_name = self.formated_nodes[pr]
                pairs += [(pr_name, cl_name)]

        if mode == "full":
            self.formated_nodes["log_prob"] = "log_prob"
        elif "short" in mode:
            self.formated_nodes["log_prob"] = "lP"
        elif mode == "math":
            self.formated_nodes["log_prob"] = r"$\log(P)$"
        else:
            self.formated_nodes["log_prob"] = "log_prob"

        for pr in self._layers[0]:
            pairs += [(self.formated_nodes[pr], r"$\log(P)$")]

        self.reverse_formated_nodes = {}
        for name, fmt_name in self.formated_nodes.items():
            self._logger.Assert(
                fmt_name not in self.reverse_formated_nodes,
                KeyError(f"Formated name {fmt_name} appears twice!"),
            )
            self.reverse_formated_nodes[fmt_name] = name

        self._logger.debug(
            "Pairs are:" + "\n".join(f"{pr} -> {cl}" for pr, cl in pairs)
        )
        return pairs

    @staticmethod
    def _voffset(ls):
        if isinstance(ls, list):
            n = len(ls)
        else:
            n = ls

        if n % 2 == 0:
            return n // 2 - 0.5
        else:
            return n // 2

    def _getNodesProps(self, fmt_nodes):

        pos = {
            self.formated_nodes[n]: [0, i - self._voffset(self._variables_layer)]
            for i, n in enumerate(self._variables_layer)
        }

        for i, layer in enumerate(self._layers[::-1]):
            for j, n in enumerate(layer):
                pos[self.formated_nodes[n]] = [
                    i + 1,
                    j - self._voffset(layer),
                ]

        pos[self.formated_nodes["log_prob"]] = [i + 2, 0]

        pos = {k: np.array(v) for k, v in pos.items()}

        nodes = [self.reverse_formated_nodes[n] for n in fmt_nodes]
        colors = []
        for fn, n in zip(fmt_nodes, nodes):
            colors.append(
                "#f8333c"
                if n == "log_prob"
                else (
                    "#fcab10"
                    if n in self._variables_layer
                    else ("#44af69" if n in self._layers[0] else "#2b9eb3")
                )
            )

        return pos, colors

    def plot(self, mode="math", n_ite=5_000, rate=5, fontsize=10):
        graph = nx.DiGraph()
        graph.add_edges_from(self._getFormatedLinks(mode))

        pos, colors = self._getNodesProps(graph.nodes)

        n_inter = np.inf
        acc, pacc, md = [], [], []

        best_n_inter = np.inf
        best_pos = pos
        best_ind = 0

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        # Return true if line segments AB and CD intersect
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        max_size_layer = max(
            [len(lay) for lay in self._layers + [self._variables_layer]]
        )

        f_variables = [self.formated_nodes[n] for n in self._variables_layer]
        f_layers = [[self.formated_nodes[n] for n in layer] for layer in self._layers]

        def trans_swap(new_pos, flay):
            a, b = sample(flay, 2)
            new_pos[b], new_pos[a] = new_pos[a], new_pos[b]

        def trans_shift(new_pos, flay):
            sel = sample(flay, 1)[0]
            y = new_pos[sel][1]
            other_y = [new_pos[fn][1] for fn in flay if fn != sel]

            if (y + 1 in other_y) and (y - 1 in other_y):
                return

            shift = 2 * np.random.randint(2) - 1
            new_y = np.clip(y + shift, -max_size_layer / 2, max_size_layer / 2)

            if len(other_y) == 0:
                new_pos[sel][1] = new_y
            elif min([abs(new_y - other) for other in other_y]) >= 1:
                new_pos[sel][1] = new_y
            else:
                new_y = np.clip(y - shift, -max_size_layer / 2, max_size_layer / 2)

                if min([abs(new_y - other) for other in other_y]) >= 1:
                    new_pos[sel][1] = new_y

        for i in trange(n_ite, desc="Building graph"):
            temp = 100 * np.exp(-rate * i / n_ite)

            new_pos = pos.copy()

            ilay = np.random.randint(-1, len(self._layers))
            if ilay == -1:
                lay = self._variables_layer
                flay = f_variables
            else:
                lay = self._layers[ilay]
                flay = f_layers[ilay]

            trans = 1  # np.random.randint(2)
            if len(lay) == 1:
                pass
            # trans_shift(new_pos, flay)
            elif len(lay) == max_size_layer:
                trans_swap(new_pos, flay)
            elif trans == 0:
                trans_shift(new_pos, flay)
            else:
                trans_swap(new_pos, flay)

            new_n_inter = np.sum(
                [
                    np.sum(
                        [
                            int(
                                intersect(
                                    new_pos[a], new_pos[b], new_pos[c], new_pos[d]
                                )
                            )
                            for a, b in graph.edges
                            if a not in (c, d) and b not in (c, d)
                        ]
                    )
                    for c, d in graph.edges
                ]
            )

            if new_n_inter == n_inter:
                max_dist = np.mean(
                    [np.linalg.norm(pos[a] - pos[b]) for a, b in graph.edges]
                )
                new_max_dist = np.mean(
                    [np.linalg.norm(new_pos[a] - new_pos[b]) for a, b in graph.edges]
                )

                p = np.exp(-(new_max_dist - max_dist) / temp)
            else:
                p = np.exp(-(new_n_inter - n_inter) / temp)
            p = min([1, p])

            if np.random.rand() < p:
                pos = new_pos
                n_inter = new_n_inter
                acc.append(1)
            else:
                acc.append(0)

            if n_inter <= best_n_inter:
                best_n_inter = n_inter
                best_pos = pos
                best_ind = i

            pacc.append(p)
            md.append(n_inter)

        if best_n_inter < n_inter:
            pos = best_pos
            self._logger.info(
                f"Setting best positions of nodes from iteration {best_ind}"
            )
        else:
            self._logger.info(
                f"Setting best positions of nodes from last iteration {best_pos}"
            )

        fig, axs = plt.subplots(nrows=3, sharex=True)
        x = np.arange(n_ite)
        axs[0].plot(x, np.cumsum(acc), c="r")
        axs[0].set_ylabel("nb accepted steps")

        axs[1].plot(x, md, c="g")
        axs[1].set_ylabel("nb intersections")

        axs[2].plot(x, pacc, c="b")
        axs[2].plot(x, np.cumsum(pacc) / (1 + x), c="r")
        axs[2].set_ylabel("P(step)")
        axs[2].set_xlabel("n")

        fig.savefig(f"{self._io.debug_dir}/graph_{mode}_convergence.pdf")

        # ext_layers = [["log_prob"]] + self._layers + [self._variables_layer]
        # ext_layers = [[self.formated_nodes[n] for n in layer] for layer in ext_layers]
        # pos = nx.shell_layout(graph, ext_layers, center=[0, 0])

        s, r = 1.5, 1
        h = s * max_size_layer
        w = s / r * (3 + len(self._layers))

        fig, ax = plt.subplots(figsize=(w, h), dpi=200)
        ax.axis("off")

        for a, b in graph.edges:
            p, q = pos[a], pos[b]
            r = 0.5 * q - 0.5 * p
            ax.plot([p[0], q[0]], [p[1], q[1]], c="k", lw=1)
            # ax.arrow(*p, *r, color="k", width=0.01)

        for c, fn in zip(colors, graph.nodes):
            ax.text(
                *pos[fn],
                fn,
                ha="center",
                va="center",
                fontsize=fontsize,
                bbox=dict(facecolor="w", edgecolor=c, boxstyle="round, pad=1"),
            )

        fig.savefig(f"{self._io.debug_dir}/graph_{mode}.pdf")
        raise
