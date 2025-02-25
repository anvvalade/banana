from os.path import join

from typing import Union, Any, Callable, Optional, List, Dict, Tuple
import inspect
from collections import OrderedDict

from jax import jit, jvp, jacfwd, jacrev
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .common import fdtype, cdtype, idtype, jnp_ndarray, Jnp_ndarray
from .logger import Logger
from .utils import (
    toList,
    f_cast,
    i_cast,
    getKeysFromDicts,
    getJaxRandomKey,
    printDict,
    updateDictRecursive,
    descriptiveStatistics,
    chainSummaryStatistics,
    printType,
)


class NotModifiableError(Exception):
    """ """

    def __init__(self, attr):
        super().__init__(
            f"Attribute {attr} should be static and should not be redefined at run-time"
        )


class AbstractMethodError(Exception):
    """ """

    def __init__(self, method_name):
        super().__init__(
            f"Method {method_name} should be abstract and should not never be called (but redefined upon inheritence"
        )


class BaseClass:
    """
    A very abtract class. Why are you reading that?
    You shouldn't be there

    Essentially creating the logger, defining the getKeysFromDicts and that's it.
    """

    init_args_aliases = {}
    _save_exts = ["npy"]
    _plot_exts = ["pdf"]

    def __init__(self, verbosity):
        self._logger = Logger(self.__class__.__name__, verbosity)

    @classmethod
    def isValidClass(cls, verbosity: int = 2):
        pass

    @classmethod
    def postLoadHook(cls, anal_dir, debug_dir, res_dir, verbosity: int = 2):
        """
        Setting the directories for outputs
        """
        cls._anal_dir = anal_dir
        cls._debug_dir = debug_dir
        cls._res_dir = res_dir

    def postInstanciationHook(self):
        pass

    def getKeysFromDicts(self, *keys, **kwargs):
        """

        Parameters
        ----------
        *keys :

        **kwargs :


        Returns
        -------


        """
        try:
            del kwargs["logger"]
        except KeyError:
            pass
        return getKeysFromDicts(*keys, logger=self._logger, **kwargs)

    def debugAssign(self, value, name):
        """replace a = b by a = self.debugAssign(b, 'a')"""
        self._logger.debug(f"Assigning {name} = {value}")

        return value

    @classmethod
    def _getSignature(
        cls,
        method: Callable,
        method_name: str,
        logger: Logger,
    ):
        """
        Low level wrapper of inspect. Returns the signature of the method:
            - args (names)
            - types (as str)
            - types (as types)
            - defaults ('no_default' for no speficied default value, i.e. inspect._empty)
        """
        # Getting the signature
        sig = inspect.signature(method)
        # init_args = [p for p in sig.parameters if p not in ignore]
        logger.debug(f"Full signature of {method_name} is {sig}")

        args, defaults, str_types, types = [], [], [], []

        for p in sig.parameters:
            # <'util.blabla'> -> blabla
            annotation = sig.parameters[p].annotation
            pr_annotation = printType(annotation, embedded=True)
            str_annotation = pr_annotation[pr_annotation.rfind(".") + 1 :]  # type: ignore

            default = sig.parameters[p].default

            # logger.debug(
            #     f"local arg: {p} has annotation {str_annotation} and default value {default}"
            # )

            args.append(p)
            defaults.append("no_default" if default is inspect._empty else default)
            str_types.append(str_annotation)
            types.append(annotation)

        return args, str_types, types, defaults

    @classmethod
    def getArgs(
        cls,
        method_name: str,
        verbosity: int,
        ignore: List[str] = ["self", "verbosity"],
    ):
        """
        High level wrapper of inspect. Returns the arguments of the method:



        """
        logger = Logger(f"{cls.__name__}:_getArgs", verbosity)

        # Checking if method exists
        logger.Assert(
            hasattr(cls, method_name),
            AttributeError(f"{method_name} is not a known method"),
        )
        method = getattr(cls, method_name)
        lookup_method_name = method_name.replace("_", "")

        # f(**kwargs) -> f(**f_args) where f_args is specified
        # allows setting up the signature before call
        args_name = f"{lookup_method_name}_args"
        if len(args := getattr(cls, args_name, [])):
            logger.debug(
                f"Found specified args for method {method_name}:" f"{', '.join(args)}"
            )
        else:
            logger.debug(f"Found no explicit arguments for method {method_name}")

        # Looking for aliases for the args
        aliases_name = f"{lookup_method_name}_args_aliases"
        if len(spec_aliases := getattr(cls, aliases_name, {})):
            logger.debug(
                f"Found aliases for method {method_name}:\n"
                f"{printDict(spec_aliases, embedded=True)}"
            )
        else:
            logger.debug(f"Found no aliases for method {method_name}")

        # Looking for default values
        defaults_name = f"{lookup_method_name}_args_defaults"
        if len(spec_defaults := getattr(cls, defaults_name, {})):
            logger.debug(
                f"Found defaults for method {method_name}:\n"
                f"{printDict(spec_defaults, embedded=True)}"
            )
        else:
            logger.debug(f"Found no defaults for method {method_name}")

        # Looking for types specifications
        types_name = f"{lookup_method_name}_args_types"
        if len(spec_types := getattr(cls, types_name, {})):
            logger.debug(
                f"Found types for method {method_name}:\n"
                f"{printDict(spec_types, embedded=True)}"
            )
            # dont know how to make a type from its string
            logger.warn("Type checking is not done for explicitly provided types")
        else:
            logger.debug(f"Found no types for method {method_name}")

        if not len(args):
            args, str_types, types, defaults = cls._getSignature(
                method, method_name, logger
            )
        else:
            defaults = ["no_default" for _ in args]
            str_types = ["no_type" for _ in args]
            types = [Any for _ in args]

        # Overwriting signature with specified values
        aliases = args.copy()
        for i, arg in enumerate(args):
            if (alias := spec_aliases.get(arg, "no_alias")) != "no_alias":
                aliases[i] = alias
            if (default := spec_defaults.get(arg, "no_default")) != "no_default":
                defaults[i] = default
            if (_type := spec_types.get(arg, "no_type")) != "no_type":
                str_types[i] = _type
                types[i] = Any

        # Ignoring some args
        # Reverse iteration to pop
        for i in range(len(args) - 1, -1, -1):
            arg = args[i]
            if arg in ignore:
                for li in [args, aliases, str_types, types, defaults]:
                    li.pop(i)

        return args, aliases, str_types, types, defaults

    # save plot
    def _savePlot(self, fig, io_dir, output_file, exts=None):
        """
        Save the plot in the proper directory with the proper format
        """
        exts = self._plot_exts if exts is None else np.atleast_1d(exts)

        self._logger.assertIsIn(
            io_dir, "io_dir", ["anal", "debug"], "available_io_dirs"
        )
        base_path = f"{getattr(self, f'_{io_dir}_dir')}/{output_file}"

        for ext in exts:
            path = f"{base_path}.{ext}"
            self._logger.debug(f"Saving {output_file} to {path}")
            fig.savefig(path)

    # save data
    def _saveData(
        self,
        data: Union[Jnp_ndarray, np.ndarray, Dict[str, Union[Jnp_ndarray, np.ndarray]]],
        io_dir: str,
        output_file: str,
        exts: Optional[List[str]] = None,
        fmt: str = "float32",
    ) -> None:
        """
        Save the data in the proper directory with the proper format
        """

        self._logger.assertIsIn(
            io_dir, "io_dir", ["anal", "debug", "res"], "available_io_dirs"
        )
        base_path = f"{getattr(self, f'_{io_dir}_dir')}/{output_file}"

        exts = self._save_exts if exts is None else np.atleast_1d(exts)

        if isinstance(data, dict):
            self._logger.Assert(
                len(exts) == 1
                and exts[0] == "npz"
                and isinstance(fmt, dict)
                and all(k in fmt for k in data),
                ValueError(
                    "dict data can only be saved to npz file, "
                    "fmt has to be a dict with the same keys as data. "
                    f"Got {exts=}, fmt={fmt} and data={printDict(data, embedded=True)}"
                ),
            )
            data = {k: np.array(v).astype(fmt[k]) for k, v in data.items()}

            path = f"{base_path}.npz"
            self._logger.debug(f"Saving dictionnary {output_file} to {path}")
            np.savez(path, **data)
            return

        data = np.array(data).astype(fmt)
        for ext in exts:
            path = f"{base_path}.{ext}"
            self._logger.debug(f"Saving {output_file}[{fmt}] to {path}")
            if ext == "bin":
                data.tofile(path)
            elif ext == "npy":
                np.save(path, data)


class BaseVariable(BaseClass):
    """
    Defines a variable (parameter) of the parameters' space.

    Contains the name of the provided quantity, how to generate random samples of it to init the
    monte carlo chain (sample the prior in some sense, although the actual prior function is a BaseModule).
    It also contains the (diagonal of the) prior std.

    Minimal example:

        class MyVariable:
            # hard coded
            provides = 'quantity_1'
            # can be defined in init
            bijector = tfb.Exp()

            def __init__(
                    self,
                    param_1: int,
                    param_2: np.ndarray,
                    my_util: MyUtil
                ):
                    self.output_size = ...
                    self.prior_std = ...
                    # define generator or overload generateInitialState()
                    self.init_state_gen = ...

            def generateInitialState(self, seed):
                # overload of define init_state_gen
                ...
                return initial_state # jnp_ndarray

    """

    provides = ""
    math_provides = ""
    input_size = None
    output_size = None

    provides_free = None
    provides_fixed = None
    provides_fixed_values = []
    provides_fixed_indices = None

    prior_mean = None
    prior_std = None
    diag_mass_matrix = None

    bijector_name = None
    bijector_kwargs = None
    bijector = None

    init_state_gen = None

    debug_output_dir = "./"

    def __init__(
        self,
        verbosity: int = 2,
    ) -> None:
        super().__init__(verbosity)

    @classmethod
    def isValidClass(cls, verbosity: int = 2):
        """
        Called right after the class is loaded by BananaCore.

        Do not touch!
        """
        logger = Logger(cls.__name__, verbosity)

        logger.assertIsInstance(cls.provides, "provides", Union[str, List[str]])

        logger.assertIsInstance(
            cls.provides_free, "provides_free", Optional[Union[str, List[str]]]
        )
        logger.assertIsInstance(
            cls.provides_fixed, "provides_fixed", Optional[Union[str, List[str]]]
        )

    @classmethod
    def postLoadHook(cls, anal_dir, debug_dir, res_dir, verbosity: int = 2) -> None:
        """
        Called when the class is loaded, after reading the attributes from the config file but
        before the initialization

        Do not touch!
        """
        super().postLoadHook(
            anal_dir,
            debug_dir,
            res_dir,
        )

        logger = Logger(cls.__name__, verbosity)

        # building provides, provides_free and provides_fixed
        cls.provides = toList(cls.provides)

        if cls.provides_free is None and cls.provides_fixed is None:
            logger.debug("Neither provides_free nor provides_fixed are set")
            cls.provides_free = cls.provides
            cls.provides_fixed = []
        elif cls.provides_free is not None and cls.provides_fixed is None:
            logger.debug(f"provides_free is set to {cls.provides_free}")
            cls.provides_free = toList(cls.provides_free)
            logger.assertIsIn(
                cls.provides_free,
                "provides_free",
                cls.provides,
                "provides",
            )
            cls.provides_fixed = [p for p in cls.provides if p not in cls.provides_free]
        elif cls.provides_free is None and cls.provides_fixed is not None:
            logger.debug(f"provides_fixed is set to {cls.provides_fixed}")
            cls.provides_fixed = toList(cls.provides_fixed)
            logger.assertIsIn(
                cls.provides_fixed,
                cls.provides_fixed,
                cls.provides,
                "provides",
            )
            cls.provides_free = [p for p in cls.provides if p not in cls.provides_fixed]
        elif not (
            all(p in cls.provides for p in cls.provides_free)
            and all(p in cls.provides for p in cls.provides_free)
        ) or any(p in cls.provides_fixed for p in cls.provides_free):
            logger.error(
                ValueError(
                    f"provides_free = {cls.provides_free} and "
                    f"provides_fixed = {cls.provides_fixed} are not consistent!"
                )
            )

    def postInstanciationHook(self) -> None:
        """
        Called after the initialization

        Do not touch!
        """
        super().postInstanciationHook()
        self._logger.assertIsNot(
            self.generateInitialState,
            "generateInitialState",
            BaseVariable.generateInitialState,
        )

        # Setting up input size and fixed values
        if self.provides_fixed != []:
            self._logger.assertIsInstance(
                self.provides_fixed_values,
                "provides_fixed_values",
                jnp_ndarray,
                length=len(self.provides_fixed),
            )

            self.provides_fixed_indices = i_cast(
                [self.provides.index(p) for p in self.provides_fixed]
            )
            self.output_size = len(self.provides)
            self.input_size = len(self.provides_free)  # type: ignore
        else:
            # input size is output size
            self.input_size = self.output_size

        self._logger.assertIsInstance(self.output_size, "output_size", int)
        self._logger.assertIsInstance(self.input_size, "input_size", int)

        self._logger.assertIsInstance(self.prior_mean, "prior_mean", jnp_ndarray)
        self._logger.assertIsInstance(self.prior_std, "prior_std", jnp_ndarray)

        self._logger.assertShapesMatch(
            input_size=[self.input_size],  # type: ignore
            prior_std=self.prior_std.shape,  # type: ignore
        )

        self._logger.debug(f"Output size is {self.output_size}")

        if self.bijector_name not in (None, ""):
            self._logger.assertIsInstance(self.bijector_name, "bijector_name", str)
            kwargs = {} if self.bijector_kwargs is None else self.bijector_kwargs
            self._logger.assertIsInstance(kwargs, "bijector_kwargs", Dict[str, Any])
            try:
                self.bijector = getattr(tfb, self.bijector_name)(**kwargs)
            except Exception as e:
                raise ImportError(
                    f"Could not load bijector {self.bijector_name}"
                    f"with kwargs {printDict(kwargs, embedded=True)}, got "
                    + "|".join(e.args)
                )

    def setPriorMeanStdAndMM(self, mean, std, dmm=None):
        for what, value in zip(["mean", "std", "dmm"], [mean, std, dmm]):
            if what == "dmm" and value is None:
                self._logger.debug("diag_mass_matrix inherited from prior_std")
                self.diag_mass_matrix = self.prior_std
                continue
            elif what == "dmm":
                nattr = "diag_mass_matrix"
            else:
                nattr = f"prior_{what}"

            attr = getattr(self, nattr)
            if attr is None:
                self._logger.debug(f"{what} set dynamically")
                setattr(self, nattr, f_cast(value))
            else:
                self._logger.info(f"{what} set statically")
                if isinstance(attr, str):
                    self._logger.Assert(
                        attr[-4:] == ".npy",
                        ValueError(f"{what} file should be npy, got {attr}"),
                    )
                    try:
                        self._logger.debug(f"Loading static {what} from {attr}")
                        setattr(self, nattr, f_cast(np.load(attr)))
                    except Exception as e:
                        self._logger.error(
                            Exception((f"Could not load static {what}: ",) + e.args)
                        )
                else:
                    try:
                        setattr(self, nattr, f_cast(attr))
                    except Exception as e:
                        self._logger.error(
                            Exception((f"Could not load static {what}: ",) + e.args)
                        )
                self._logger.Assert(
                    getattr(self, nattr).ndim == 1,  # type: ignore
                    ValueError(
                        f"{what} should be of dimension 1 "
                        "(full std / mass matrices not handled yet, use bijectors)"
                    ),
                )

                self._saveData(
                    np.array(getattr(self, nattr)),
                    "res",
                    f"{what}_{self.provides}",
                    exts=["npy"],
                )

    def generateInitialState(self, seed: int) -> Jnp_ndarray:
        """

        Parameters
        ----------
        seed: int :


        Returns
        -------


        """
        if self.init_state_gen is not None:
            try:
                return self.init_state_gen.sample(seed=getJaxRandomKey(seed))
            except Exception as e:
                self._logger.error(e)
        else:
            self._logger.error(AbstractMethodError("generateInitialState"))

    # Wrapped fixed outputs

    def wrapStateToDict(self, varstate):
        if len(self.provides) == 1:
            return {self.provides[0]: varstate}
        elif len(self.provides) == varstate.size:
            return {p: jnp.atleast_1d(v) for p, v in zip(self.provides, varstate)}
        elif len(self.provides) > varstate.size:
            state = jnp.insert(
                varstate,
                self.provides_fixed_indices,
                self.provides_fixed_values,
                axis=-1,
            )
            return {p: jnp.atleast_1d(v) for p, v in zip(self.provides, state)}

        else:
            self._logger.error(
                ValueError(
                    f"Cannot wrap state of size {varstate.size} to "
                    f"{self.provides} of size {len(self.provides)}"
                )
            )

    # Summarize

    @staticmethod
    def _summaryBinning(data, order, bins_boundaries):
        data_st = data[order]
        return np.array(
            [
                np.mean(data_st[le:ri])
                for le, ri in zip(bins_boundaries[:-1], bins_boundaries[1:])
            ]
        )

    def _getSummaryBins(self, plot_vs, plot_vs_func):
        if not hasattr(self, "_summary_bins"):
            self._summary_plot_vs = (
                np.arange(self.output_size)  # type: ignore
                if (isinstance(plot_vs, str) and plot_vs == "indices")
                else plot_vs
            )  # type: ignore
            if plot_vs_func in ("loglog", "semilogx"):
                self._summary_plot_rat_func = "semilogx"
                lpvs = np.log10(self._summary_plot_vs)
                self._summary_bins = np.logspace(
                    np.min(lpvs) * 0.99, np.max(lpvs) * 1.01, 100
                )
            else:
                self._summary_plot_rat_func = "plot"
                _min = np.min(self._summary_plot_vs)
                _max = np.max(self._summary_plot_vs)
                _min = 0.99 * _min if _min > 0 else 1.01 * _min
                _max = 0.99 * _max if _max < 0 else 1.01 * _max
                self._summary_bins = np.linspace(_min, _max, 100)
            self._summary_x = (self._summary_bins[1:] * self._summary_bins[:-1]) ** 0.5

            self._summary_order = np.argsort(self._summary_plot_vs)
            self._summary_plot_vs_sorted = self._summary_plot_vs[self._summary_order]
            self._summary_bins_boundaries = np.searchsorted(
                self._summary_plot_vs_sorted, self._summary_bins
            )

            self._summary_pr_mean_binned = self._summaryBinning(self.prior_mean, self._summary_order, self._summary_bins_boundaries)  # type: ignore
            self._summary_pr_std_binned = self._summaryBinning(self.prior_std, self._summary_order, self._summary_bins_boundaries)  # type: ignore

        # So that they can be used in the "_summary" namespace
        return (
            self._summary_order,
            self._summary_bins_boundaries,
            self._summary_pr_mean_binned,
            self._summary_pr_std_binned,
            self._summary_x,
            self._summary_plot_rat_func,
        )

    def _plotSummary(
        self,
        name_run,
        quantity,
        what,
        plot_vs_func,
        plot_vs_label,
        order,
        bins_boundaries,
        x,
        post,
        label,
        pr_binned=None,
        ratio=None,
        plot_rat_func=None,
    ):
        fig, axs = plt.subplots(nrows=1 + int(ratio is not None), sharex=True)
        axs = np.atleast_1d(axs)

        po_binned = self._summaryBinning(post, order, bins_boundaries)

        self._logger.Assert(
            hasattr(axs[0], plot_vs_func),
            AttributeError(f"Cannot use plot_vs_func={plot_vs_func}"),
        )
        pfunc = getattr(axs[0], plot_vs_func)

        if pr_binned is not None:
            pfunc(x, pr_binned, "k", label="prior")
        pfunc(x, po_binned, "r", label=label)

        fig.suptitle(f"{quantity} ({what}, {name_run})")
        fig.legend(ncols=2, loc="upper right")

        axs[0].set_ylabel(f"{what}")

        if ratio is not None:
            ra_binned = self._summaryBinning(ratio, order, bins_boundaries)
            pfunc_rat = getattr(axs[1], plot_rat_func)  # type: ignore
            axs[1].axhline(1, linestyle=":", color="k")
            pfunc_rat(x, ra_binned, "r")
            axs[1].set_ylim(0, None)
            axs[1].set_ylabel("Posterior / Prior")

        axs[-1].set_xlabel(plot_vs_label)
        self._savePlot(fig, "anal", f"{name_run}_{what}_{quantity}")

    def summarize(
        self,
        chain,
        name_run,
        plot_vs="indices",
        plot_vs_func="plot",
        plot_vs_label="indices",
    ) -> None:
        quantity = "_".join(self.provides_free)  # type: ignore
        self._logger.debug(f"Summarizing {self.__class__.__name__}:{quantity}")

        # simple summary
        if chain.ndim == 1:
            self._logger.descriptiveStatistics(chain, quantity)
            return

        # Function in utils, has to be jitted, could be used in other contexts too
        post_mean, post_std, ratio_mean, ratio_std, post_signi, r_split, chain_ess = (
            chainSummaryStatistics(
                chain,
                self.prior_mean,  # type: ignore
                self.prior_std,  # type: ignore
                ess=chain.size <= 128**3 * 100,
            )
        )

        for what, data in zip(
            ["mean", "std", "significance", "Rsplit", "ESS"],
            [post_mean, post_std, post_signi, r_split, chain_ess],
        ):
            self._saveData(
                data,
                "res",
                f"{name_run}_{what}_{quantity}",
                exts=["npy"],
            )

        if self.output_size < 10 or len(self.provides_free) > 1:  # type: ignore
            if self.input_size == 1:
                rows = self.provides
            else:
                rows = (
                    [f"output {i}" for i in range(self.input_size)]  # type: ignore
                    # if self.provides_free is None
                    # else self.provides_free
                )
            self._logger.printTable(
                rows=rows,  # type: ignore
                columns=[
                    "post mean",
                    "post std",
                    "post std / prior std",
                    "(po_m - pr_m) / po_std",
                    "R split",
                    "ESS",
                    "ESS / sample size (%)",
                ],
                data=np.stack(
                    [
                        post_mean,
                        post_std,
                        ratio_std,
                        post_signi,
                        r_split,
                        chain_ess,
                        100 * chain_ess / chain.shape[0],
                    ]
                ).T,
            )
        else:
            # if output_size is bigger or outputs are not named, we make plots
            # showing mean, cov, post_signi and ess
            self._logger.descriptiveStatisticsTable(
                level="info",
                title=f"Summary of {name_run}:{quantity}",
                names=dict(
                    pm="post mean",
                    ps="post std",
                    rs="post std / prior std",
                    psi="(po_m - pr_m) / po_std",
                    rsplit="R Split",
                    ess="ESS",
                    ess_rate="ESS / sample size (%)",
                ),
                pm=post_mean,
                ps=post_std,
                rs=ratio_std,
                psi=post_signi,
                rsplit=r_split,
                ess=chain_ess,
                ess_rate=100 * chain_ess / chain.shape[0],
            )

            (
                order,
                bins_boundaries,
                pr_mean_binned,
                pr_std_binned,
                x,
                plot_rat_func,
            ) = self._getSummaryBins(plot_vs, plot_vs_func)

            ess_prior = chain.shape[0] * jnp.ones_like(chain_ess)
            for what, post, pr_binned, rat, label in zip(
                ["mean", "std", "significance", "Rsplit", "ESS"],
                [post_mean, post_std, post_signi, r_split, chain_ess],
                [
                    pr_mean_binned,
                    pr_std_binned,
                    None,
                    jnp.ones_like(x),
                    jnp.ones_like(x),
                ],
                [ratio_mean, ratio_std, None, None, chain_ess / ess_prior],
                [
                    "posterior",
                    "posterior",
                    # r"$\frac{\big<{\rm post}\big> - \big<{\rm prior}\big>}{\sigma_{\rm post}}$",
                    "significance",
                    "rsplit",
                    "ess",
                ],
            ):
                self._plotSummary(
                    name_run,
                    quantity,
                    what,
                    plot_vs_func,
                    plot_vs_label,
                    order,
                    bins_boundaries,
                    x,
                    post,
                    label=label,
                    pr_binned=pr_binned,
                    ratio=rat,
                    plot_rat_func=plot_rat_func,
                )


class BaseModule(BaseClass):
    """
    Inner items of the calling tree.
    Takes (some) quantities, as provided by variables or other modules and returns ONE quantity.

    Minimal example:

        class MyQuantity3:
            provides = 'quantity_3'

            def call(self, quantity_1, quantity_2):
                # do stuff
                ...
                # return the results
                # name of the returned value is ignored
                return quantity_3

    This class takes quantity_1 and quantity_2 provided by other variables or modules and returns
    quantity_3, which can then be used by another module. The returned value needs to be a
    differentiable jax.numpy.array (jnp_array).

    The signature is interpreted to build the tree. Arguments names need to be "provides" values of other variables or
    modules.

    Four methods have to (can) be overloaded:
        - call: needs to be as efficient as possible
        - debug: needs to be as verbose as possible (can save and plot data)
        - details: returns a lot of inner variables that can be useful for analysis / debugging
        - timed: for profiling (not ready yet)

    The instanciation is done automatically by BananaCore at run time.
    The signature (parameters AND types) are interpreted:

        class MyQuantity3:
            def __init__(
                self,
                param_1: int,
                param_2: np.ndarray,
                local_util_name: MyUtil
            ):
                pass

    searches for the parameters (param_1 & 2) in the defined parameters and data (keys). Helper
    classes "utils" can also be provided as arguments (see BaseUtil). The 'type hint' is used to define the class
    to pass.

    Only one instance of each module is created within each run. Static attributes can be modified
    from the configuration:

        modules:
            MyQuantity3:
                static_attribute_1: 1
                static_attribute_2: 'custom'

    The default is:

        modules:
            MyQuantity3: ~


    """

    debug_do = [
        "call_overloadeddebug_method",
        "check_returned_dtype",
        "check_returned_shape",
        "check_returned_value",
        "show_returned_value_stats",
        # "check_gradients", # broken
        # "show_gradients_stats",
        # "check_jacobians",
        # "save_jacobians",
        # "show_jacobians_stats",
    ]
    debug_output_dir = "./"
    debug_verbosity = 2

    provides = ""
    math_provides = ""

    call_args_aliases = {}
    call_args_fixed = {}

    def __init__(
        self,
        verbosity: int = 2,
    ) -> None:
        super().__init__(verbosity)

    @classmethod
    def isValidClass(cls, verbosity: int = 2):
        logger = Logger(cls.__name__, verbosity)

        # duck typing
        logger.assertIsInstance(cls.provides, "provides", Union[str, List[str]])
        logger.assertIsNot(cls.call, "call", BaseModule.call)

    @classmethod
    def postLoadHook(cls, anal_dir, debug_dir, res_dir, verbosity: int = 2) -> None:
        """
        Called when the class is loaded, after reading the attributes from the config file but
        before the initialization

        Do not touch!
        """
        super().postLoadHook(
            anal_dir,
            debug_dir,
            res_dir,
        )

        logger = Logger(cls.__name__, verbosity)

        cls.provides = toList(cls.provides)
        sig = inspect.signature(cls.call)

        # building requires
        fixed = list(cls.call_args_fixed.keys())
        cls.call_args = [p for p in sig.parameters if p != "self"]
        cls.call_args_free = [p for p in cls.call_args if p not in fixed]

        if (aliases := getattr(cls, "call_args_aliases", None)) is None:
            cls.requires = cls.call_args_free
        else:
            cls.requires = [aliases.get(arg, arg) for arg in cls.call_args_free]  # type: ignore

        logger.Assert(
            "log_prob" not in cls.requires, ValueError("Module cannot require log_prob")
        )

        # Checking signatures match
        for meth_name in ["debug", "details", "timed"]:
            method = getattr(cls, meth_name)
            if method is not getattr(BaseModule, meth_name):
                sig = list(inspect.signature(method).parameters)[1:]
                logger.Assert(
                    sig == cls.call_args,
                    ValueError(
                        f"Signature of {meth_name} ({', '.join(sig)}) "
                        f"does not match with the signature of call "
                        f"({', '.join(cls.call_args)})"
                    ),
                )

    def postInstanciationHook(self) -> None:
        """ """
        super().postInstanciationHook()

    def call(self):
        """
        To be overloaded: the core function, what does this module do?

        The signature of the function is used to construct the calling-tree:

            def call(self, a, b, c):
                return a + b + c

        will search for modules / variables that provide "a, b, c".
        Providers will be called upstream.

        TODO: handle optional values
        """
        self._logger.error(AbstractMethodError("call"))

    def details(self, *args):
        """
        To be overloaded: returns are dict(key, values) of "inner" / "intermediate"
        values that could be used for debug, analysis, etc.

        Takes the same args as call.

        TODO: check the naming convention for modules that return log_prob (to avoid clashes)
        """
        quantity = (
            f"{self.__class__.__name__}_log_prob"
            if self.provides[0] == "log_prob"
            else self.provides[0]
        )
        return {quantity: self.call(*args)}

    def debug(self, *args):
        """
        To be overloaded: put a lot of check ups in there,
        and save data and plots with self._saveData and self._savePlot.

        Takes the same args are call.

        Default value: returns call
        """
        return self.call(*args)

    def timed(self, *args):
        """
        To be overloaded: profiling, returns as dict(name: time).

        NOT READY TO USE
        """
        return {"total": self.call(*args)}

    def _wrapArgs(self, *args, method_name="call", fixed_values=None):
        """
        Wrap fixed and free args together to create the good signature for call / debug, etc
        """
        fixed_values = (
            getattr(self, f"{method_name}_args_fixed")
            if fixed_values is None
            else fixed_values
        )
        _args = []
        i = 0
        for arg in getattr(self, f"{method_name}_args"):
            if arg in fixed_values:
                _args.append(fixed_values[arg])
            else:
                _args.append(args[i])
                i += 1

        return _args

    def getWrapped(self, method_name, args_method_name=None, return_dict=True):
        """
        Return the properly wrapped method, with (possibly) fixed arguments
        """
        args_method_name = method_name if args_method_name is None else args_method_name
        if return_dict:

            def wrapped(*args):
                res = getattr(self, method_name)(
                    *self._wrapArgs(*args, method_name=args_method_name)
                )
                if isinstance(res, tuple):
                    self._logger.Assert(
                        len(self.provides) == len(res),
                        ValueError(
                            f"provides has length {len(self.provides)} "
                            f"while returned value has length {len(res)}"
                        ),
                    )
                    return dict(zip(self.provides, res))
                elif isinstance(res, dict):
                    return res
                else:
                    return {self.provides[0]: res}

            return wrapped

        else:
            return lambda *args: getattr(self, method_name)(
                *self._wrapArgs(*args, method_name=args_method_name)
            )

    def extendedDebug(self, *args):
        """
        A wrapper around the debug function to automate some sanity checks.
        Takes the same args as call (or debug).
        """

        logger = Logger(
            f"{self.__class__.__name__}debug_log_prob", self.debug_verbosity
        )
        logger.debug(f"Entering debug call with debug args {self.debug_do}")
        if (
            "call_overloadeddebug_method" in self.debug_do
            and self.__class__.debug is not BaseModule.debug
        ):
            debug_ret = self.debug(*args)
        else:
            debug_ret = None
        returned_values = self.call(*args)

        rv_is_tuple = isinstance(returned_values, tuple)
        rv_as_tuple = returned_values if rv_is_tuple else (returned_values,)
        dr_as_tuple = debug_ret if rv_is_tuple else (debug_ret,)

        if rv_is_tuple:
            logger.Assert(
                len(returned_values) == len(self.provides),
                ValueError(
                    f"call returned a tuple of size {len(returned_values)}: "
                    f"while provides has len {len(self.provides)}: {self.provides}"
                ),
            )
        else:
            logger.Assert(
                len(self.provides) == 1,
                ValueError(
                    f"provides = {self.provides} has length {len(self.provides)}"
                    "while returned value is not a tuple"
                ),
            )

        if debug_ret is not None:
            if rv_is_tuple:
                logger.Assert(
                    (len(returned_values) == len(debug_ret)),
                    ValueError(
                        f"Debug returned value: \n{debug_ret} \ndoes "
                        f"not match with call returned value: \n{returned_values}"  # type: ignore
                    ),
                )
            logger.Assert(
                all(
                    [
                        jnp.allclose(rv, dr, atol=np.inf, rtol=1e-3)  # type: ignore
                        for rv, dr in zip(rv_as_tuple, dr_as_tuple)
                    ]
                ),
                ValueError(
                    f"Debug returned value: \n{debug_ret} \ndoes "
                    f"not match with call returned value: \n{returned_values}"  # type: ignore
                ),
            )

        if "check_returned_dtype" in self.debug_do:
            # Avoiding throwing an error when the clash is just between numpy array and jax array
            # There is a better way to do that, no doubt
            #
            # print(returned_value.dtype)
            # if returned_value.dtype == np.dtypes.Float32DType and fdtype == jnp.float32:
            #     pass
            # else:
            #     logger.assertIs(returned_value.dtype, "returned value dtype", fdtype)
            for i, rv in enumerate(rv_as_tuple):
                ret_t = str(rv.dtype)  # type: ignore
                logger.Assert(
                    "float" in ret_t, ValueError("Returned type should be float")
                )
                is_f32 = "32" in str(fdtype)
                logger.Assert(
                    (is_f32 and "32" in ret_t) or (not is_f32 and "64" in ret_t),
                    ValueError(
                        f"Returned type of {i}th value should be float"
                        + ("32" if is_f32 else "64")
                        + f", got instead {fdtype}"
                    ),
                )

        if "check_returned_shape" in self.debug_do:

            logger.info(f"Returned shape is {(rv.shape for rv in rv_as_tuple)}")  # type: ignore
            if self.provides[0] == "log_prob":
                logger.Assert(
                    not rv_is_tuple and returned_values.shape == (),  # type: ignore
                    ValueError("Module should return log_prob of shape ()"),
                )

        if "check_returned_value" in self.debug_do:
            for i, (rv, pr) in enumerate(zip(rv_as_tuple, self.provides)):
                stats, err = descriptiveStatistics(rv, mod=jnp, return_err=True)  # type: ignore
                if err:
                    jnp_attr = {}
                    for key in dir(self):
                        val = getattr(self, key)
                        if isinstance(val, jnp_ndarray):
                            jnp_attr[key] = val
                    logger.info(
                        f"{self.__class__.__name__} has jax attributes \n{printDict(jnp_attr, embedded=True)}"
                    )
                    logger.error(
                        ValueError(
                            f"Computed '{pr}' with shape "
                            f"{rv.shape} and statistics \n{stats}"  # type: ignore
                        ),
                    )
                logger.info(
                    f"Computed '{pr}' with shape "
                    f"{rv.shape} and statistics \n{stats}"  # type: ignore
                )

        if "check_gradients" in self.debug_do:
            logger.checkVJP(self.call, args, self.__class__.__name__)
        logger.success("My work here is done!")
        return returned_values


class BaseUtil(BaseClass):
    """
    Helper classes, instanciated once and shared between all variables, modules, analyses and other
    utils.

    The inner structure as well as the methods are not constrained.

    Only the signature of the __init__ is constained, to support automatic instanciation:

        class MyUtil(BaseUtil):
            def __init__(self, a:float, b:int, c:OtherUtil):
                pass

    See BaseModule for more on the signature.

    This class is empty for now, and just used as a "tag" to be loaded by BananaCore.
    """

    pass


class BaseKernel(BaseClass):
    """
    Wrappers of the tensorflow_probability kernels, with predefined traces functions.

    The prior std is built from the variables.
    If variables have non-default bijectors, a TransformedKernel is created.

    See tensorflow_probability for more details about the kwargs

    Check up BABANADIR/base_library/kernels.py for default kernels.
    """

    trace_return = []

    def __init__(
        self,
        varinstances_sorted: List[BaseVariable],
        log_prob_fn: Callable[[Jnp_ndarray], Jnp_ndarray],
        verbosity: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(verbosity)

        self.varinstances_sorted = varinstances_sorted
        self._log_prob_fn = log_prob_fn
        self._mass_matrix = [
            jnp.atleast_1d(varinst.prior_std) for varinst in self.varinstances_sorted  # type: ignore
        ]

        self.__dict__.update(
            updateDictRecursive(
                self.getKernelKwargs(), kwargs, logger=self._logger, inplace=False
            )
        )
        self._logger.debug(
            f"At the end of init, kernel kwargs is \n{printDict(self.getKernelKwargs(), embedded=True)}"
        )

        self.buildBijectors()

    def postInstanciationHook(self):
        """
        Duck typing, no concession
        """
        self._logger.assertIsNot(self.trace, "trace", BaseKernel.trace)
        self._logger.Assert(
            self.trace_return != [],
            ValueError("trace_return is empty"),
        )
        self._logger.assertIsNot(self.getKernel, "getKernel", BaseKernel.getKernel)

    def getKernelKwargs(self):
        """ """
        return {k: getattr(self, k) for k in dir(self) if "kwargs" in k}

    def trace(self, chain_state, previous_kernel_result) -> Tuple[jnp_ndarray]:
        """
        Custom hard-coded trace
        """
        self._logger.error(AbstractMethodError("trace"))

    def getKernel(self):
        """ """
        self._logger.error(AbstractMethodError("kernel"))

    def buildBijectors(self):
        """
        Builds the list of bijectors for the (possibly) transformed transition kernel.
        See maybeTransformKernel.

        """
        self.bijectors = []
        for varinst in self.varinstances_sorted:
            if varinst.bijector is None:
                self.bijectors.append(tfb.Identity())
            else:
                self._logger.debug(
                    f"{varinst.__class__.__name__} has a bijector: {varinst.bijector}"
                )
                self.bijectors.append(varinst.bijector)

    def maybeTransformKernel(self, kernel):
        """
        Possibly wraps the kernel into a TransformedTransitionKernel, if
        the `variable.bijector` attribute is provided in (at least one of) the variables.

        This should be used only for 'root' samplers, like HMC, NUTS or MALA, not for step size
        adaptation or else.
        """

        if len(self.bijectors):
            self._logger.info("Wrapping kernel into a transition kernel")
            return tfp.mcmc.TransformedTransitionKernel(kernel, self.bijectors)
        else:
            return kernel

    @staticmethod
    def _getInnermostResults(prev_kern_res):
        if hasattr(prev_kern_res, "inner_results"):
            return BaseKernel._getInnermostResults(prev_kern_res.inner_results)
        else:
            return prev_kern_res

    @classmethod
    def summarize(cls, trace: List[np.ndarray], name_run) -> None:
        """
        Custom summary. Default uses the "trace_return" keys for a simple prompt.

        Needs to be a class method to be called without instance in an analysis context.
        """
        to_print = ""
        for trace_res, name in zip(trace, cls.trace_return):
            _max, _min, _mean, _std = [
                float(getattr(np, func)(trace_res))
                for func in ["max", "min", "mean", "std"]
            ]
            _first = float(trace_res[0])
            _last = float(trace_res[-1])
            if np.all((trace_res == 0) + (trace_res == 1)):
                to_print += (
                    f"{name:<20s}: "
                    f"rate = {_mean:11.2%} | first = {_first:10.0f} | last = {_last:12.0f}\n"
                )
            else:
                to_print += (
                    f"{name:<20s}: max = {_max:12.2f} | min = {_min:12.2f} | "
                    f"mean = {_mean:12.2f} | std = {_std:12.2f} | "
                    f"first = {_first:12.2f} | last = {_last:12.2f}\n"
                )

        # We build the logger just for one line to be consistent with the variables summaries
        Logger(cls.__name__, verbosity=2).info(f"Summary of {name_run}\n{to_print}")

    @classmethod
    def analyze(
        cls,
        trace: List[np.ndarray],
        time_kernel_run: float,
        name_run: str,
        output_dir: str,
    ) -> None:
        """
        Custom analyse of the trace. Default uses the "trace_return" keys for a simple prompt.
        TODO: second sentence above is a lie, default does nothing.
        """

        n_tr = len(trace)
        fig, axs = plt.subplots(nrows=n_tr, sharex=True, figsize=(8, 3 * n_tr))
        axs = np.atleast_1d(axs)

        x = np.arange(1, len(trace[0]) + 1)
        for i, (ax, tr, tr_ret) in enumerate(zip(axs, trace, cls.trace_return)):
            if np.all(tr > 0):
                ax.semilogy(x, tr, c=f"C{i}")
            ax.plot(x, tr, c=f"C{i}")
            # ax.plot(x, (np.cumsum(tr[::-1]) / x)[::-1], c="k", ls=":")
            ax.plot(x, gaussian_filter(tr, x.size // 20), c="k", ls=":")
            ax.set_ylabel(tr_ret)

        axs[-1].set_xlabel("MC step")

        fig.savefig(f"{output_dir}/{name_run}_trace.pdf")


class BaseStrategy(BaseClass):
    """
    A series of commands to run.

    Heavily based on dictionnaries.
    Quite hard to understand, you're very deep in the code (too deep?).

    Built and called in BananaCore.runStrategy(), see this function for more.

    See BANANADIR/base_library/strategies.py for working precoded strategies.
    """

    # names = []
    # commands = []
    cmd_kwargs = {}

    def __init__(self, verbosity: int = 2, **cmd_kwargs: Any):
        super().__init__(verbosity)
        updateDictRecursive(
            self.cmd_kwargs,
            cmd_kwargs,
            logger=self._logger,
            limit_to_existing_keys=False,
        )

        # THAT HAS TO BE CHANGED!
        # BERK BERK BERK
        for cmd, val in self.cmd_kwargs.items():
            command = self.commands[cmd]
            if command in (
                "saveResults",
                "summarize",
                "forgetChain",
                "runTraceAnalysis",
                "runAnalysis",
                "saveDetails",
            ):
                # we dont want to forget the time per step
                if command != "forgetChain":
                    val.update({"time_per_step": None})

                name_run = getKeysFromDicts(
                    "name_run",
                    cmd_kwargs_val=val,
                    logger=self._logger,
                )
                self._logger.assertIsNotNone(name_run, "name_run")
                self._logger.debug(
                    f"Adding wildcard '{name_run}:*' to cmd_kwargs[{cmd}]"
                )
                val.update({f"{name_run}:*": None})

        self.cmd_counter = 0
        self._logger.debug(
            f"At the end of init, commands kwargs is \n{printDict(self.cmd_kwargs, embedded=True)}"
        )

    def __iter__(self):
        self.cmd_counter = 0
        return self

    def __next__(self):
        try:
            name = self.names[self.cmd_counter]
        except IndexError:
            raise StopIteration
        command = self.commands[name]
        kwargs = self.cmd_kwargs[name]
        self._logger.debug(
            f"Command {name} runs {command} with default kwargs \n{printDict(kwargs, embedded=True)}"
        )
        self.cmd_counter += 1
        return command, kwargs

    @classmethod
    def isValidClass(cls, verbosity: int = 2):
        logger = Logger(cls.__name__, verbosity)
        logger.assertIsInstance(cls.names, "names", List[str])
        logger.assertIsInstance(cls.commands, "commands", Dict[str, str])
        logger.assertIsInstance(
            cls.cmd_kwargs,
            "cmd_kwargs",
            Dict[str, Dict[str, Any]],
            debug=True,
        )
        for name in cls.names:
            logger.assertIsIn(name, name, cls.commands, "commands")
            logger.assertIsIn(name, name, cls.cmd_kwargs, "cmd_kwargs")
        logger.assertShapesMatch(
            cmd_kwargs=[len(cls.cmd_kwargs)],
            names=[len(cls.names)],
            commands=[len(cls.commands)],
        )

    @property
    def names(self):
        """ """
        return self.names

    @names.setter
    def names(self, _):
        """
        Making it unmodifiable in the init
        """
        self._logger.error(
            NotModifiableError("names"),
        )

    @property
    def commands(self):
        """ """
        return self.commands

    @commands.setter
    def commands(self, _):
        """
        Making it unmodifiable in the init
        """
        self._logger.error(
            NotModifiableError("commands"),
        )


class BaseAnalysis(BaseClass):
    """
    A class to run analyses on the data, after the chain has run.

    Instanciated automatically alongside the variables and the modules, see BaseModule for the form of
    the __init__ signature.

    Two functions have to be overloaded:
        - oneState, called on each results of the detailsLogProb on each state of the chain
        - finalize, called once after the whole chain has run
    See these functions for more details.

    Set save_level > 0 to set the "verbose" level of the saving.
    The higher save_level is, the more data is saved.
    Same for plot_level > 0.

    Data is saved in the results directory. Plots are saved in the analysis directory.

    You can set the extensions of the saving (bin, npy) and plotting (pdf, jpeg, etc) from the
    configuration file:
        analysis:
            MyAnalysis:
                _save_exts = ['npy']
                _save_exts = ['pdf']

    """

    _output_dir: Optional[str] = None
    _results_output_dir: Optional[str] = None
    save_level: Optional[int] = None
    plot_level: Optional[int] = None

    def __init__(
        self,
        verbosity: int = 2,
    ) -> None:
        super().__init__(verbosity)

    # called by core
    def prepare(
        self,
        plotting_output_dir: str,
        results_output_dir: str,
        save_level: Optional[int],
        plot_level: Optional[int],
    ):
        """
        Setting up attributes at run time
        If [save, plot]_level are None, the hard-coded versions are used
        """
        self._plotting_output_dir = plotting_output_dir
        self._results_output_dir = results_output_dir
        self.save_level = save_level if save_level is not None else self.save_level
        self.plot_level = plot_level if plot_level is not None else self.plot_level

    # main functions, have to be over-written
    def oneState(
        self,
        real: int,
    ) -> None:
        """
        Called on the output of details = detailsLogProb(state) for each state of the chain
        Signature is used to chose which item(s) of details are needed, for instance:

            def oneState(self, divv_modes):
                pass

            receives `divv_modes` as argument at each iteration.

        Signature should contain only values returned by detailsLogProb. There is no check before
        calling the function, so it might fail!

        TODO: handle optional arguments
        """
        self._logger.debug("oneState has not been overloaded: doing nothing")

    def finalize(
        self,
    ) -> None:
        """
        Called on the output of details = detailsLogProb(state) for the WHOLE chain
        Signature is used to chose which item(s) of details are needed, for instance:

            def finalize(self, divv_modes):
                pass

            receives the WHOLE CHAIN of `divv_modes` as argument.

        Signature should contain only values returned by detailsLogProb. There is no check before
        calling the function, so it might fail!

        For large arguments, please prefer "oneState" over "finalize", as it avoids storing large
        data.

        TODO: handle optional arguments
        """
        self._logger.debug("finalize has not been overloaded: doing nothing")


"""

    # @property
    # def provides(self) -> str:
        # return self.provides

    # @provides.setter
    # def provides(self, _):
        # self._logger.error(NotModifiableError("provides"),)

    # @property
    # def output_shape(self) -> List[List[int]]:
        # return self._output_shape

    # @output_shape.setter
    # def output_shape(self, shape: List[List[int]]) -> None:
        # self._logger.assertIsInstance(
            # shape, "output_shape", list, instance_items=int, length=1
        # )
        # self._output_shape = shape

    # @property
    # def prior_std(self) -> Jnp_ndarray:
        # return self._diag_std_matrix

    # @prior_std.setter
    # def prior_std(self, dcm: Jnp_ndarray) -> None:
        # self._logger.Assert(
            # isinstance(dcm, jnp_ndarray) and dcm.dtype == dtype,
            # f"Diag prior std {dcm} must be a jnp_ndarray(dtype={dtype})",
        # )
        # self._logger.Assert(
            # dcm.shape == self.output_shape,
            # f"Diag prior std of shape {dcm.shape} is not consistent "
            # "with output shape {self.output_shape}",
        # )
        # self._diag_std_matrix = dcm

    # @property
    # def limits(self) -> [Optional[float], Optional[float]]:
        # return self.limits

    # @limits.setter
    # def limits(self, limits: [Optional[float], Optional[float]]) -> None:
        # self._logger.Assert(
            # all(isinstance(lim, (float, type(None))) for lim in limits),
            # f"Limits {limits} should be of type float",
        # )
        # if limits[0] is not None and limits[1] is not None:
            # self._logger.Assert(
                # limits[0] < limits[1], f"Limits {limits} are not sorted"
            # )
        # self.limits = limits

    # @property
    # def needs_instances_at_init(self) -> List[str]:
        # return self.needs_instances_at_init
    
    # @needs_instances_at_init.setter
    # def needs_instances_at_init(self, _) -> None:
        # self._logger.error(NotModifiableError("needs_instances_at_init"),)





    # @property
    # def provides(self) -> str:
        # return self.provides

    # @provides.setter
    # def provides(self, _):
        # self._logger.error(
            # "Trying to redefine provides at run time",
            # NotModifiableError("provides"),
        # )

    # @property
    # def requires(self) -> List[str]:
        # return self.requires

    # @requires.setter
    # def requires(self, _):
        # self._logger.error(NotModifiableError("requires"),)

    # @property
    # def optional_requires(self) -> List[str]:
        # return self.optional_requires

    # @optional_requires.setter
    # def optional_requires(self, _):
        # self._logger.error(NotModifiableError("optional_requires"),)

    # @property
    # def needs_instances_at_init(self) -> List[str]:
        # return self.needs_instances_at_init

    # @needs_instances_at_init.setter
    # def needs_instances_at_init(self, _):
        # self._logger.error(NotModifiableError("needs_instances_at_init"),)

    # @property
    # def output_shape(self) -> List[List[int]]:
        # return self._output_shape

    # @output_shape.setter
    # def output_shape(self, shape: List[List[int]]) -> None:
        # self._logger.Assert(
            # len(shape) == len(self.requires),
            # f"output shape {shape} not compatible "
            # f"with provided quantity len {self.provides}",
        # )
        # self._logger.Assert(
            # all(isinstance(ln, int) for ln in shape),
            # f"Elements of output_shape {shape} are not all int",
        # )
        # self._output_shape = shape

    # @property
    # def input_shape(self) -> List[List[int]]:
        # return self._input_shape

    # @input_shape.setter
    # def input_shape(self, shape: List[List[int]]) -> None:
        # self._logger.Assert(
            # len(shape) == len(self.requires),
            # f"Input shape {shape} not compatible "
            # f"with required quantities len {self.requires}",
        # )
        # self._logger.Assert(
            # all(isinstance(ln, int) for ln in shape),
            # f"Elements of input_shape {shape} are not all int",
        # )
        # self._input_shape = shape
"""
