from importlib.machinery import SourceFileLoader
import inspect
from os import listdir
from os.path import join
import time

from tqdm import trange, tqdm
from typing import Union, Any, List, Optional, Dict, Callable, Tuple

from jax import jit
import jax
import jax.numpy as jnp
from jax.test_util import check_grads
from tensorflow_probability.substrates import jax as tfp

# from simple_optax import minimize as optax_minimize
# from infsuite.blackjax import run_nuts

import numpy as np
import matplotlib.pyplot as plt

from .common import banana_path, Jnp_ndarray, Jnp_ndarray, fdtype
from .baseclasses import (
    BaseModule,
    BaseVariable,
    BaseKernel,
    BaseStrategy,
    BaseUtil,
    BaseClass,
    BaseAnalysis,
)
from .tree import Tree
from .logger import Logger
from .io import IO
from .utils import (
    f_cast,
    descriptiveStatistics,
    updateDictRecursive,
    getKeysFromDicts,
    getJaxRandomKey,
    printDict,
    time2hms,
    printType,
    isInstance,
    atLeast2DLastAxis,
)


def getParentClasses(_cls):
    """
    List all the parent classes

    Parameters
    ----------
    _cls : input class


    Returns
    -------
    classes : list
              parent classes of _cls


    """
    parents = list(_cls.__bases__)
    for parent in parents:
        if parent == object:
            break
        else:
            return parents + getParentClasses(parent)
    return [parent for parent in parents if parent != object]


class BananaCore:
    """
    This is the core object of banana.
    It allows to build a complex posterior distribution and run MC like methods on it.
    It manages the inputs and outputs.
    For an extended docstring, ask me!
    """

    _welcome = (
        "####################################\n"
        "###                              ###\n"
        "###       Welcome to Banana      ###\n"
        "###                              ###\n"
        "###    Have fun exploring the    ###\n"
        "###  ungodly highly dimensional  ###\n"
        "###       parameters space       ###\n"
        "###                              ###\n"
        "####################################\n"
    )

    # general
    _description = ""
    _working_dir = ""
    _base_libraries_dir = f"{banana_path}/base_library"
    _libraries_dirs = []

    # logging
    _verbosity = 2
    _debug_classes = "none"

    # stategy
    _strategy = ""
    _strategy_init_kwargs = {}

    # parameters / data
    _parameters = {}
    _data = {}

    # modnames / varnames / analnames / utilnames
    _selected_varnames_initattr = {}
    _selected_modnames_initattr = {}
    _selected_analnames_initattr = {}
    _selected_utilnames_initattr = {}
    _typeclasses_typenames = dict(
        mod="modules",
        var="variables",
        anal="analyses",
        ker="kernels",
        strat="strategies",
        util="utils",
    )

    # loading stuff
    _library_files = []
    _all_loaded_classnames_classes = {}
    _all_loaded_classnames_libfiles = {}
    _all_loaded_provided_by_classnames = {}
    _all_loaded_modnames = []
    _all_loaded_varnames = []
    _all_loaded_utilnames = []
    _all_loaded_analnames = []
    _all_loaded_kernames = []
    _all_loaded_stratnames = []

    # instanciating it
    _classnames_instances = {}

    # tree
    _layers = []
    _links = {}
    _variables_layer_sorted = []
    _inner_layers_sorted = []
    _last_layer = []
    _max_layers = 10
    _max_depth_instanciation = 10
    _tree_args = {}

    # INIT AND LOAD

    def __init__(
        self,
        config: Union[dict, str],
        argv: List[str],
        verbosity: int = 2,
    ):
        # First, be polite
        print(self._welcome)

        # Get the logging running
        # No working dir for now
        self._logger = Logger("Core", verbosity)

        # All the initialization is delegated to IO to keep things clean here
        # check of the configuration, some recasting, initializing the working dir
        # and the logger...
        self._io = IO(verbosity=verbosity)
        self.__dict__.update(self._io.initCore(config, argv))

        # Overwrite the verbosity
        self._logger._verbosity = self._verbosity

        self._logger.debug(
            f"Core attributes are {printDict(self.__dict__, embedded=True)}"
        )

        # A visual check that the good config is being ran
        if self._description != "":
            desc = f"### {self._description} ###"
            desc = "#" * len(desc) + f"\n{desc}\n" + "#" * len(desc)
            self._logger.info("Run described as: \n" f"{desc}")

        if jax.lib.xla_bridge.get_backend().platform == "gpu":
            self._logger.success("Running on (at least one) GPU")
        else:
            to_print = "#############\n" "No GPU Found!\n" "#############\n"
            self._logger.warn(to_print)

    # GETTERS

    def _getKeysFromDicts(self, *keys: str, **kwargs: Any) -> Any:
        """
        See `utils.py` -> getKeysFromDicts for more detail
        """
        return getKeysFromDicts(*keys, logger=self._logger, **kwargs)

    def _getClassVerbosity(self, classname: str) -> int:
        """
        if self._debug_classes == 'all' or classname in self._debug_classes:
            return 3
        else:
            return 2
        """
        # weird order but faster to check
        if self._debug_classes == "all":
            return 3
        elif self._debug_classes == "none":
            return 2
        elif classname in self._debug_classes:
            return 3
        else:
            return 2

    def _getClassHash(self, _class) -> int:
        """
        Returns the hash of the class source code.
        Used to check if the class has changed when reloading a model.
        """
        return hash(inspect.getsource(_class))

    def _getClassesOrInstancesAux(self, names, dic, namedic, err_msg, return_type):
        """
        Searches for keys "names" in dictionary "self.dic".
        Raises ValueError with "err_msg" if key is not Found
        Return_type is either "dict", "list" or "auto".
        If "auto", it returns a list if there is more than one value or list[0] if len(list) == 1
        """
        try:
            dic = getattr(self, dic)
        except KeyError:
            self._logger.error(error=ValueError(err_msg))
        else:
            names = [names] if isinstance(names, str) else names
            ret = self._getKeysFromDicts(*names, **{namedic: dic}, return_dict=True)
            if return_type == "dict":
                return ret
            elif len(ret) == 1 and return_type == "auto":
                return list(ret.values())[0]
            elif return_type in ("list", "auto"):
                # dict dont always preserve the order
                return [ret[n] for n in names]
            else:
                raise ValueError(
                    f"Return type = {return_type} should be type, list or auto"
                )

    def getClasses(self, names: Union[List[str], str], return_type: str = "auto"):
        """
        Get loaded classes from their names.
        Return_type is either "dict", "list" or "auto".
        If "auto", it returns a list if there is more than one value or list[0] if len(list) == 1
        """
        return self._getClassesOrInstancesAux(
            names,
            "_all_loaded_classnames_classes",
            "loaded_classes",
            "Classes have not been loaded yet",
            return_type=return_type,
        )

    def getInstances(
        self, names: Union[List[str], str], return_type: str = "auto"
    ) -> Union[Dict[str, Any], List[Any], Any]:
        """
        Get instances from their names.
        Return_type is either "dict", "list" or "auto".
        If "auto", it returns a list if there is more than one value or list[0] if len(list) == 1
        """
        return self._getClassesOrInstancesAux(
            names,
            "_classnames_instances",
            "instances",
            "Classes have not been instanciated yet",
            return_type=return_type,
        )

    # LOAD LIBRARIES

    # Hook on modules / variables / analyses

    def _applyConfigToBaseClass(self, typeclass: str, typename: str) -> None:
        all_loaded = getattr(self, f"_all_loaded_{typeclass}names")
        try:
            selected = getattr(self, f"_selected_{typeclass}names_initattr")
        except AttributeError:
            # There is no selection possible on this type of class
            return

        self._logger.info(f"Applying config to {typename}")
        if not len(selected):
            self._logger.debug(
                f"No class of type {typename} has been explicitely selected"
            )
            return

        self._logger.assertIsIn(
            list(selected.keys()),
            list(selected.keys()),
            all_loaded,
            f"all loaded {typename}",
        )

        for classname, classattr in selected.items():
            _class = self._all_loaded_classnames_classes[classname]
            if classattr is None:
                continue
            # try:
            #     ref_hash = classattr.pop("hash")
            # except KeyError:
            #     pass
            # else:
            #     current_hash = self._getClassHash(_class)
            #     if ref_hash != current_hash:
            #         self._logger.warn(
            #             f"Hash of {classname} has changed between stored version and stored run"
            #         )

            for k, v in classattr.items():
                setattr(_class, k, v)

            if len(classattr):
                mod_dict = {k: v for k, v in _class.__dict__.items() if k in classattr}
                self._logger.info(
                    f"Static attributes were set on {classname}:\n{printDict(mod_dict, embedded=True)}"
                )

    def _buildLibraryFiles(
        self,
    ) -> None:
        """
        Selects the files that will be inspected within the given directories.
        Only python files ending with ".py" and containing the keywords
            variables, modules, kernels, strategies, utils, analyzes
        are parsed.
        The remaining files are ignored.

        If files are explicitely provided, they are added to the list, even if they do not match the
        selection criterions above.

        The private attribute _libraries_dirs is updated.

        Parameters
        ----------
        libraries_dirs: List[str] : explicitely provided directories


        Returns
        -------
        None

        """
        self._logger.info("Setting and loading libraries")
        self._libraries_dirs += [self._base_libraries_dir]
        self._logger.debug(
            "Libraries dirs to explore:\n" + "\n".join(self._libraries_dirs)
        )

        self._library_files = []
        for ld in self._libraries_dirs:
            for f in listdir(ld):
                # Check on type
                # Avoiding doubles
                if (
                    f[-3:] == ".py"
                    and (
                        "modules" in f
                        or "variables" in f
                        or "strategies" in f
                        or "kernels" in f
                        or "utils" in f
                        or "analyses" in f
                    )
                    and join(ld, f) not in self._library_files
                ):
                    self._library_files.append(join(ld, f))
                else:
                    self._logger.debug(f"Ignoring file {join(ld, f)}")

    def _loadOneClass(self, libfile, name, _cls):
        # Keeping only the classes that are defined in this file
        # and not imports
        if libfile not in str(_cls):
            return

        # Check if there a parent is a BaseClass
        parents = getParentClasses(_cls)
        self._logger.debug(f"Trying to load {name} ({_cls})")
        self._logger.indent()
        self._logger.debug(f"{name} has parents {parents}")
        if BaseVariable in parents:
            typeclass = "var"
        elif BaseModule in parents:
            typeclass = "mod"
        elif BaseUtil in parents:
            typeclass = "util"
        elif BaseAnalysis in parents:
            typeclass = "anal"
        elif BaseKernel in parents:
            typeclass = "ker"
        elif BaseStrategy in parents:
            typeclass = "strat"
        else:
            self._logger.debug(
                f"{name} from file {libfile} is ignored (not based on Base[...])",
            )
            self._logger.unindent()
            return

        # A first duck-typing check
        try:
            self._logger.debug(f"{name} running post load hooks")
            new_ind = self._logger.indent()
            _ = _cls.isValidClass(self._verbosity)
            self._logger.setIndent(new_ind - 1)
        except Exception as e:
            err = (
                f"{name} from file {libfile} did not pass sanity check!"
                + "\nError was\n"
                + "|".join(e.args)
                + f"\n{name} has attributes:"
                + "\n       "
                + "\n       ".join(
                    [f"{key}: {str(val)}" for key, val in _cls.__dict__.items()]
                ),
            )
            self._logger.error(
                ImportError(err),
            )
        # Avoid doubles
        self._logger.debug(f"Looking for {name} in already loaded names")
        if name in self._all_loaded_classnames_libfiles:
            self._logger.error(
                ImportError(
                    f"{name} defined several times. "
                    f"It was found (at least in) {libfile} and "
                    f"{self._all_loaded_classnames_libfiles[name]}"
                )
            )

        self._logger.unindent()

        # Storing class
        self._all_loaded_classnames_classes[name] = _cls
        self._all_loaded_classnames_libfiles[name] = libfile
        getattr(self, f"_all_loaded_{typeclass}names").append(name)

    def _loadLibraries(self) -> None:
        """
        Parses the files of the _library_files attribute and search for classes that inherit from
        the classes defined in the `baseclasses.py` file.

        Explicitely provided classes are added.

        A series of attributes containing the available variables, modules, etc are created and updated.

        Parameters
        ----------
        explicitly_provided_classes: List[Any] :
                                     Explicitely provided classes


        Returns
        -------
        None

        """
        self._buildLibraryFiles()
        self._logger.setIndent(0)
        self._logger.debug(
            "Library files to load:\n    " + "\n    ".join(self._library_files)
        )

        # Some maps:
        # name <> classe
        self._all_loaded_classnames_classes = {}
        # name <> source file
        self._all_loaded_classnames_libfiles = {}
        # quantities <> class name
        self._all_loaded_provided_by_classnames = {}
        # all classes that are [modules, variables, kernels, strategies, util, analysis]
        self._all_loaded_modnames = []
        self._all_loaded_varnames = []
        self._all_loaded_utilnames = []
        self._all_loaded_analnames = []
        self._all_loaded_kernames = []
        self._all_loaded_stratnames = []

        for libfile in self._library_files:
            self._logger.debug(f"Loading {libfile}")
            # libfile_base = libfile[libfile.rfind("/") + 1 :]
            lib = SourceFileLoader(libfile, libfile).load_module()
            # Loading all the classes in the file
            new_classes = dict(inspect.getmembers(lib, inspect.isclass))
            self._logger.indent()
            for name, _cls in new_classes.items():
                self._loadOneClass(libfile, name, _cls)
            self._logger.unindent()

        # if attributes are set explicitely
        for typeclass, typename in self._typeclasses_typenames.items():
            self._applyConfigToBaseClass(typeclass, typename)

        # Running the load hook (after the modification with the config)
        for classname, _class in self._all_loaded_classnames_classes.items():
            _class.postLoadHook(
                self._io.anal_dir,
                self._io.debug_dir,
                self._io.res_dir,
                self._getClassVerbosity(classname),
            )

        for typeclass, typename in self._typeclasses_typenames.items():
            try:
                selected = getattr(self, f"_selected_{typeclass}names_initattr")
            except AttributeError:
                # There is no selection possible on this type of class
                continue
            if not len(selected):
                continue
            self._logger.info(
                f"Selected {typename}:\n"
                + self.listLoadedClasses(typeclass, selected.keys(), embedded=True)  # type: ignore
            )

        # Storing all the available quantities provided by modules and variables
        for name, _class in self.getClasses(
            self._all_loaded_modnames + self._all_loaded_varnames, return_type="dict"
        ).items():  # type: ignore

            for prov in _class.provides:
                if prov in self._all_loaded_provided_by_classnames:
                    self._all_loaded_provided_by_classnames[prov] += [name]
                else:
                    self._all_loaded_provided_by_classnames[prov] = [name]

        for typeclass, typename in self._typeclasses_typenames.items():
            self._logger.debug(
                f"Loaded {typename}:\n"
                + self.listLoadedClasses(typeclass, embedded=True)  # type: ignore
            )

        self._logger.success("Loaded libraries")

    # BUILD THE LOG PROB FUNCTION

    def _createLink(self, client: str, provider: str) -> None:
        """
        Create a dependency link between a provider [variable, module] and a client [module].

        The private attribute `_links` is updated


        Parameters
        ----------
        client : str : name of the client

        provider : str : name of the provider


        Returns
        -------
        None


        """
        try:
            self._links[client].index(provider)
        except ValueError:
            # links[client] exists, but provider is not linked
            self._links[client].append(provider)
        except KeyError:
            # links[client] does not exist yet
            self._links[client] = [provider]

    def _buildNextLayer(self, previous_layer: List[str]) -> List[str]:
        """
        Build a layer of the tree given the previous layer.

        This method inspects the required quantities of the modules of the previous layers and tries to build the
        next one with the available variables / modules.

        In the first stage of the tree building, a variable / module may appear twice. It will be
        trimmed later.

        Parameters
        ----------
        previous_layer : List[str] : previous layer


        Returns
        -------
        next_layer : List[str] : next layer


        """
        current_layer = []
        # Iterating over previous layer

        for modname, _class in self.getClasses(
            previous_layer, return_type="dict"
        ).items():
            # Getting what this module wants
            # Copying it as we'll modify it
            required = _class.requires.copy()

            self._logger.debug(f"{modname} requires {required}")

            # Iterating over arguments
            self._logger.indent()
            for arg in required:

                self._logger.Assert(
                    arg in self._all_loaded_provided_by_classnames,
                    KeyError(
                        f"{arg} demanded by {modname} is not provided by any loaded class"
                    ),
                )

                # dict to calm down my linter
                providers = dict(  # type: ignore
                    self.getClasses(  # type: ignore
                        self._all_loaded_provided_by_classnames[arg], return_type="dict"
                    )
                )

                self._logger.debug(
                    f"{arg} provided by\n    " + "\n    ".join(providers.keys())  # type: ignore
                )

                # to record all the matches
                matches = []

                # there is only one match
                if len(providers) == 1:
                    prov = list(providers.keys())[0]
                    match = [
                        "var" if prov in self._all_loaded_varnames else "mod",
                        prov,
                    ]
                    matches.append(match)
                    # removing 's' at the end of typenames
                    self._logger.debug(
                        f"{self._typeclasses_typenames[match[0]][:-1]} {prov} is the only provider!"
                    )

                # oops, there is more than one, let's see...
                elif len(providers) > 1:
                    self._logger.debug("There are several providers")

                    # Check if any is explicitly selected
                    # we record all the matches, in case of conflict!
                    for prov, _class in providers.items():
                        if prov in self._selected_varnames_initattr:
                            matches.append(["var", prov])
                        elif prov in self._selected_modnames_initattr:
                            matches.append(["mod", prov])

                # Not provided and mandatory or provided by more than one class
                else:
                    mods = self.listLoadedClasses(typeclass="mod", embedded=True)
                    vars = self.listLoadedClasses(typeclass="var", embedded=True)
                    self._logger.error(
                        error=ValueError(
                            f"Banana could not find the good provider for {arg}."
                            f"This quantity is required by {modname}.\n"
                            f"Matches are {matches}.\n"
                            "This can be the case because (1) either there is no module nor "
                            f"variable providing {arg}, or (2) there are "
                            f"several modules / variables providing {arg} and none was selected "
                            "explicitly.\n"
                            f"Selected variables: {list(self._selected_modnames_initattr.keys())} \n"
                            f"Selected modules: {list(self._selected_varnames_initattr.keys())} \n"
                            f"Loaded modules: \n{mods} \n"
                            f"Loaded variables: \n{vars} \n"
                        )
                    )

                # implicit else
                # we're here only if len(match) == 1
                # we avoid unnecessary indentation
                typeclass, prov = matches[0]

                # type name without the 's'
                typename = self._typeclasses_typenames[typeclass][:-1]

                self._createLink(modname, prov)
                self._available_quantities.append(arg)
                self._logger.debug(f"Linking {modname} to {typename} {prov}")
                if typeclass == "var":
                    self._variables_layer.append(prov)
                else:
                    current_layer.append(prov)

            self._logger.unindent()

        return current_layer

    def _buildTree(self):
        """
        This function builds the calling tree of all the variables and modules.

        Explicitely variables and modules are part of that tree.
        If they cannot for some reason, an error is raised.

        Not explicitely, but necessary modules can be guessed, unless there is a clash
        (i.e. two modules provide the same quantity).
        In this case, one of the clashing modules needs to be explicitely demanded.

        The tree is built recursively using the "requires" and "provides" attributes of all the
        classes, starting from the modules which provide the "log_prob" quantity.

        In order to be called in the log_prob() function, the tree is properly flattened,
        so that every module has what it needs when it is called.

        """
        self._logger.setIndent()
        self._logger.info("Building tree")
        self._layers = [
            [
                name
                for name, _class in self.getClasses(
                    list(self._selected_modnames_initattr.keys()), return_type="dict"
                ).items()  # type: ignore
                if _class.provides[0] == "log_prob"
            ]
        ]
        self._logger.debug(f"Layer 0 is: \n{self._layers[0]}")
        self._variables_layer = []
        self._links = {}
        self._available_quantities = []

        for layer in range(1, self._max_layers):
            self._logger.debug(f"Entering layer {layer}")
            self._logger.indent()
            current_layer = self._buildNextLayer(self._layers[-1])
            self._logger.unindent()
            self._logger.debug(f"Layer {layer} is {current_layer}")
            if current_layer == []:
                break
            else:
                self._layers.append(current_layer)
        else:
            self._tree = Tree(
                layers=self._layers,
                variables_layer=self._variables_layer,
                links=self._links,
                getInstances=self.getInstances,
                io=self._io,
                verbosity=self._verbosity,
                **self._tree_args,
            )
            self._logger.error(
                ValueError(
                    f"Maximal number of layers reached ({self._max_layers}). "
                    f"The current tree is {self._tree.print()}"
                )
            )

        self._logger.setIndent()
        self._logger.debug("Checking for several occurences of the same module")
        self._logger.indent()
        # not very elegant but it works -- maybe there is a better way
        # the pb is that we are modifying _layers while iterating on it.
        # probably not the best idea!
        clean = False  # enter the loop
        while not clean:
            clean = True  # Tree is clean unless an "double occurence" is found
            for i_layer, layer in enumerate(self._layers):
                if i_layer == 0:
                    continue
                for i_mod, mod in enumerate(layer):
                    self._logger.debug(
                        f"Checking for occurences of {mod} at layer {i_layer}"
                    )
                    self._logger.indent()
                    if i_mod < len(layer) - 1 and mod in layer[i_mod + 1 :]:
                        self._logger.debug(
                            f"Killing {mod} at layer {i_layer} (appeared twice in same layer)"
                        )
                        self._layers[i_layer].pop(i_mod)
                        clean = False
                    if any(
                        [
                            mod in prev_layer
                            for prev_layer in self._layers[i_layer + 1 :]
                        ]
                    ):
                        self._logger.debug(f"Killing {mod} at layer {i_layer}")
                        self._layers[i_layer].pop(self._layers[i_layer].index(mod))
                        clean = False
                    self._logger.unindent()
                    # self._logger.debug(f"Layers are: \n{self._layers}")
            if clean:
                self._logger.setIndent()
                self._logger.debug("Tree is clean!")
            else:
                self._logger.unindent()
                self._logger.debug(
                    "Maybe some work left to do, starting another cleaning loop!"
                )
                self._logger.indent()

        self._inner_layers_sorted = []
        for layer in self._layers[1:][::-1]:
            self._inner_layers_sorted += layer

        self._last_layer = self._layers[0]
        self._variables_layer = [str(item) for item in np.unique(self._variables_layer)]
        self._available_quantities = list(np.unique(self._available_quantities))

        self._tree = Tree(
            layers=self._layers,
            variables_layer=self._variables_layer,
            links=self._links,
            getClasses=self.getClasses,
            io=self._io,
            verbosity=self._verbosity,
            **self._tree_args,
        )

        self._logger.debug(
            f"Modules to run are (in this order): {self._inner_layers_sorted}"
        )
        self._logger.success("Tree built")
        self._logger.info(f"The tree is \n{self._tree.print(embedded=True)}")

        if self._verbosity > 3:
            self._tree.plot(mode="math", n_ite=7_500, rate=5, fontsize=12)

        # Not allowing unused selected variables: they're just always bugs
        for varname in self._selected_varnames_initattr:
            if varname not in self._variables_layer:
                self._logger.error(
                    error=ValueError(f"Unused selected variable: {varname}")
                )

        # Not allowing unused selected modules: they're just always bugs
        for modname in self._selected_modnames_initattr:
            if modname not in self._inner_layers_sorted + self._last_layer:
                self._logger.error(
                    error=ValueError(f"Unused selected module: {modname}")
                )

    def _instanciateOneClass(
        self,
        classname: str,
        trace: List[str] = ["root"],
    ):
        trace_str = "->".join(trace)
        self._logger.debug(
            f"Initializing {classname} from {trace_str}",
        )
        self._logger.indent()

        # Checking recursion is doing ok
        self._logger.Assert(
            classname not in trace,
            ValueError(
                "There is a circular dependency in the instanciation. "
                f"Trace is\n{trace_str}"
            ),
        )
        self._logger.Assert(
            len(trace) < self._max_depth_instanciation,
            ValueError(f"Maximal instanciation depth reached. Trace is\n{trace_str}"),
        )

        # Checking we're not trying to load something weird
        self._logger.Assert(
            classname not in self._all_loaded_kernames + self._all_loaded_stratnames,
            ValueError(
                f"Kernels and strategies cannot be used as args: trying to instanciate {classname}."
            ),
        )

        # getting class
        # We should discover a problem here, but let's be safe
        baseclass = self._getKeysFromDicts(
            classname, all_loaded_classnames=self._all_loaded_classnames_classes
        )

        self._logger.debug("Interpreting init arguments")
        self._logger.indent()

        # parsing __init__
        (locals, globals, str_types, types, defaults) = baseclass.getArgs(
            "__init__",
            verbosity=self._verbosity,
        )

        self._logger.unindent()

        # We look for loaded classnames
        # only Optional[] is accepted as class spec, not Union[], or else
        is_loaded_class = []
        _globals = []
        for i, (glob, _type) in enumerate(zip(globals, str_types)):
            for substr in ["Optional[", "]"]:
                _type = _type.replace(substr, "")
            if _type in self._all_loaded_classnames_classes:
                is_loaded_class.append(True)
                globals[i] = _type
            else:
                is_loaded_class.append(False)

        error_not_found = [default == "no_default" for default in defaults]

        # Some printing
        to_print = f"{'local':<30s} {'global':>30s} {'type':>40s} {'is loaded class':>20s} {'has_default':>15s}"
        to_print = to_print + "\n" + "-" * len(to_print) + "\n"
        for loc, glob, _type, default, is_ld_cl in zip(
            locals, globals, str_types, defaults, is_loaded_class
        ):
            _glob = glob if glob != loc else "."
            to_print += (
                f"{loc:<30s} {_glob:>30s} {_type:>40s} "
                f"{bool(is_ld_cl):>20b} {default != 'no_default':>15b}\n"
            )

        self._logger.debug("Got signature\n" + to_print)

        # getting general parameters
        args = {}
        for glob in ["parameters", "data"]:
            if glob in globals:
                ind = globals.index(glob)
                args[glob] = getattr(self, "_" + glob)
                for li in [
                    locals,
                    globals,
                    str_types,
                    types,
                    defaults,
                    error_not_found,
                    is_loaded_class,
                ]:
                    li.pop(ind)

        # we're going to do it a lot...
        def filter_loaded_classes(*lists, keep=True):
            ret = [
                [
                    item
                    for item, is_ld_cl in zip(_list, is_loaded_class)
                    if is_ld_cl == keep
                ]
                for _list in lists
            ]
            return ret[0] if len(ret) == 1 else ret

        # First normal arguments
        # Getting global parameters values
        _locals, _globals, _str_types, _types, _defaults, _errors = (
            filter_loaded_classes(
                locals, globals, str_types, types, defaults, error_not_found, keep=False
            )
        )
        try:
            glob_values = self._getKeysFromDicts(
                *_globals,
                parameters=self._parameters,
                data=self._data,
                error_not_found=False,
                not_found_return_value=_defaults,
                return_dict=True,
                recursive=True,
            )
        except KeyError as e:
            self._logger.error(
                KeyError(f"While initializing {classname}, got \n" + "|".join(e.args))
            )

        self._logger.debug(
            "Raw args from parameters / data\n" + printDict(glob_values, embedded=True)
        )

        # checking types of args
        for loc, glob, str_type, _type in zip(_locals, _globals, _str_types, _types):
            name = f"{glob} ({classname}:{loc})" if loc != glob else glob
            if str_type != "no_type" and _type != Any:
                self._logger.assertIsInstance(
                    glob_values[glob], f"{classname}:{name}", _type
                )

        # updating args
        args.update({loc: glob_values[glob] for loc, glob in zip(_locals, _globals)})

        # Then classes
        # Initializing dependencies
        _locals, _globals, _defaults, _errors = filter_loaded_classes(
            locals, globals, defaults, error_not_found, keep=True
        )
        for loc, _classname, err in zip(_locals, _globals, _errors):
            # We instanciate the class only if 'err_not_found == True'
            # else, we ignore
            if _classname in self._classnames_instances:
                pass
            elif err:
                self._instanciateOneClass(_classname, trace + [classname])
            else:
                try:
                    self._instanciateOneClass(_classname, trace + [classname])
                except Exception as e:
                    self._logger.warn(
                        f"Could not initialize optional {classname}:{_classname}, got\n"
                        + "|".join(e.args)
                    )

        # Getting instances
        instances = self._getKeysFromDicts(
            *_globals,
            instanciated_classes=self._classnames_instances,
            error_not_found=_errors,
            not_found_return_value=_defaults,
            return_dict=True,
        )

        # updating args
        args.update({loc: instances[glob] for loc, glob in zip(_locals, _globals)})

        self._logger.debug(f"Got args {printDict(args, embedded=True)}")

        # Initializing
        try:
            self._classnames_instances[classname] = baseclass(
                **args,
                verbosity=self._getClassVerbosity(classname),
            )
        except Exception as e:
            e.args = (f"While initializing {classname}: ",) + e.args
            self._logger.error(e)

        # A bit of duck-typing
        self._classnames_instances[classname].postInstanciationHook()
        self._logger.unindent()
        self._logger.success(f"Instanciated {classname} smoothly")

    def _instanciateAllClasses(
        self,
    ) -> None:
        """
        Instanciates all the classes following the determined order.

        Parameters
        ----------
        debug_classes: Union[str, List[str]]:
            Accepted values ("all", "none", "") or list of classnames to debug.

        Returns
        -------


        """

        self._logger.info(
            f"Parameters are {printDict(self._parameters, embedded=True)}"
        )
        self._logger.descriptiveStatisticsTable(
            level="info", title="Data are", **self._data
        )

        BaseModule.debug_output_dir = self._io.debug_dir
        BaseVariable.debug_output_dir = self._io.debug_dir

        classnames_to_instanciate = (
            self._variables_layer
            + self._inner_layers_sorted
            + self._last_layer
            + list(self._selected_analnames_initattr.keys())
        )

        # Actually do the instanciation
        self._classnames_instances = {}
        for classname in classnames_to_instanciate:
            if classname not in self._classnames_instances:
                self._instanciateOneClass(classname)
        self._logger.success(
            "Instanciated all classes, got some extra definitions to do now"
        )

        # Now we have everything we need to construct a "state"
        # Sorting variables by order of size of the output
        # Not sure why there is an "if" here, removing it.
        # if self._variables_layer_sorted is None:
        varsizes = [
            varinst.output_size
            for varinst in self.getInstances(self._variables_layer, return_type="list")
        ]

        _, self._variables_layer_sorted = zip(
            *sorted(zip(varsizes, self._variables_layer))
        )

        self._varinstances_sorted = self.getInstances(
            self._variables_layer_sorted, return_type="dict"
        )
        self._state_size = np.sum(
            [varinst.output_size for varinst in self._varinstances_sorted.values()]
        )

        self._log_prob_signature = [
            ", ".join(varinst.provides)
            for varinst in self._varinstances_sorted.values()
        ]
        self._logger.info(
            "Signature of the log_prob function: "
            f"log_prob({', '.join(self._log_prob_signature)})"
        )

        to_print = f"{'Name':<50} | {'Provides':<50} | Size\n"
        for varname, varinst in self._varinstances_sorted.items():
            to_print += f"{varname:<50} | {', '.join(varinst.provides):<50} | {varinst.output_size}\n"
        bytes_per_float = 4 if fdtype == "float32" else 8
        self._state_size_MB = bytes_per_float * self._state_size / 1024**2
        to_print += f"{'State size (total)':<103s} | {self._state_size} ({self._state_size_MB} MB)"
        self._logger.info(f"Details about state size \n{to_print}")

    def buildLogProb(
        self,
        load_libraries: str = "auto",
        build_tree: str = "auto",
        instanciate_all: str = "auto",
    ):
        """
        Runs all the necessary functions to build the `log_prob`. Can be called on an already
        initialized BananaCore instance (to help debug for instance).

        Each building step can be separately called. If [step] == "auto" it is rerun
        if it has never been ran or if it's expected to change because a prior step has just been
        alterated.
        """
        if load_libraries == "auto":
            if not self._all_loaded_classnames_classes:
                self._loadLibraries()
                loaded_libraries = True
            else:
                self._logger.debug("Did not overwrite loaded libraries")
                loaded_libraries = False
        elif load_libraries == "overwrite":
            loaded_libraries = True
            self._logger.warn("Overwriting loaded libraries")
            self._loadLibraries()
        elif not self._all_loaded_classnames_classes:
            self._logger.error(ValueError("Did not load libaries and none are loaded"))

        if build_tree == "auto":
            if not self._layers or loaded_libraries:
                self._buildTree()
                built_tree = True
            else:
                self._logger.debug("Did not overwrite tree")
                built_tree = False
        elif build_tree == "overwrite":
            self._logger.warn("Overwriting tree")
            self._buildTree()
            built_tree = True
        elif loaded_libraries:
            self._logger.warn("Tree was not rebuilt although libraries were reloaded")

        if instanciate_all == "auto":
            if not self._classnames_instances or loaded_libraries:
                self._instanciateAllClasses()
                instanciated_all = True
            else:
                self._logger.debug("Did not overwrite instanciation")
                instanciated_all = False
        elif instanciate_all == "overwrite":
            self._logger.warn("Overwriting instanciation")
            self._instanciateAllClasses()
            instanciated_all = True
        elif loaded_libraries:
            self._logger.warn(
                "Did not re-instanciate classes although libraries were reloaded!"
            )

        if not (loaded_libraries or built_tree or instanciated_all):
            self._logger.warn("Nothing was done!")

    # Default log prob functions

    def getLogProb(self) -> Callable:
        """
        Constructs and returns the log_prob function
        """

        def log_prob(*state: Jnp_ndarray) -> Jnp_ndarray:
            """

            Parameters
            ----------
            states : List[Jnp_ndarray]:
                List of the states (per Variable) in order of growing size

            Returns
            -------


            """
            local_quantities = {}
            for varinst, varstate in zip(self._varinstances_sorted.values(), state):
                local_quantities.update(varinst.wrapStateToDict(varstate))

            for modinst in self.getInstances(self._inner_layers_sorted):
                local_quantities.update(
                    modinst.getWrapped("call", return_dict=True)(
                        *[local_quantities[req] for req in modinst.requires]
                    ),
                )

            return f_cast(
                jnp.sum(
                    jnp.stack(
                        [
                            modinst.getWrapped("call", return_dict=False)(
                                *[local_quantities[req] for req in modinst.requires]
                            )
                            for modinst in self.getInstances(
                                self._last_layer, return_type="list"
                            )
                        ]
                    )
                )
            )

        return log_prob

    def getDetailsLogProb(
        self, keep_quantities: Optional[List[str]] = None
    ) -> Callable:
        def details_log_prob(*state: Jnp_ndarray) -> Dict[str, Jnp_ndarray]:
            """

            Parameters
            ----------
            state: Jnp_ndarray :


            Returns
            -------


            """
            local_quantities = {}
            for varinst, varstate in zip(self._varinstances_sorted.values(), state):  # type: ignore
                local_quantities.update(varinst.wrapStateToDict(varstate))

            for modinst in self.getInstances(self._inner_layers_sorted):
                # print(
                #     [local_quantities[req] for req in modinst.requires]  # type: ignore
                # )
                local_quantities.update(
                    modinst.getWrapped("details", args_method_name="call", return_dict=True)(  # type: ignore
                        *[
                            local_quantities[req] for req in modinst.requires  # type: ignore
                        ]
                    ),
                )

            log_prob = 0.0
            for modname, modinst in self.getInstances(
                self._last_layer, return_type="dict"
            ).items():  # type: ignore
                ret = modinst.getWrapped(
                    "details", args_method_name="call", return_dict=False
                )(*[local_quantities[req] for req in modinst.requires])
                local_quantities.update(ret)
                log_prob = log_prob + ret[f"{modname}_log_prob"]

            local_quantities["log_prob"] = log_prob

            if keep_quantities is None:
                return local_quantities
            else:
                return {
                    q: v for q, v in local_quantities.items() if q in keep_quantities
                }

        return details_log_prob

    def getDebugLogProb(self):
        def debug_log_prob(
            *state: Jnp_ndarray,
            debug_classes: Union[str, List[str]] = self._debug_classes,
        ) -> float:
            """

            Parameters
            ----------
            state :
                A state

            debug_classes: Union[str, List[str]]:
                Accepted values ("all", "none", "") or list of classnames to debug in `debug_lob_prob`

            Returns
            -------


            """
            logger = Logger("Core:debug_log_prob", verbosity=BaseModule.debug_verbosity)
            # Prepare debug_classes
            if debug_classes == "all":
                debug_classes = list(self._classnames_instances.keys())
            elif debug_classes in ("none", ""):
                debug_classes = []
            else:
                logger.assertIsInstance(debug_classes, "debug_classes", List[str])

            local_quantities = {}
            logger.info("Slicing state...")
            self._logger.assertShapesMatch(
                state=[len(state)], expected=[len(self._varinstances_sorted)]
            )
            for (varname, varinst), varstate in zip(
                self._varinstances_sorted.items(), state  # type: ignore
            ):
                if varname in debug_classes:
                    logger.info(
                        f"{varname} gets {varinst.provides} of shape "
                        f"{varinst.output_size} "
                    )
                    logger.info(
                        f"{varname} has fixed variables "
                        f"{varinst.provides_fixed} with values "
                        f"{varinst.provides_fixed_values}"
                    )
                local_quantities.update(varinst.wrapStateToDict(varstate))
                if varname in debug_classes:
                    # varinst.summarize(np.atleast_2d(varstate))
                    logger.descriptiveStatistics(varstate, varname)

            for modname, modinst in self.getInstances(
                self._inner_layers_sorted, return_type="dict"
            ).items():  # type: ignore
                meth = "call"
                if modname in debug_classes:
                    logger.info(
                        f"{modname} computes {modinst.provides} from [{', '.join(modinst.requires)}]"
                    )
                    meth = "extendedDebug"

                local_quantities.update(
                    modinst.getWrapped(meth, args_method_name="call", return_dict=True)(
                        *[local_quantities[req] for req in modinst.requires]
                    )
                )

            log_prob = 0.0
            for modname, modinst in self.getInstances(
                self._last_layer, return_type="dict"
            ).items():  # type: ignore
                meth = "call"
                if modname in debug_classes:
                    logger.info(
                        f"{modname} computes {modinst.provides} from {modinst.requires}"
                    )
                    meth = "extendedDebug"
                log_prob = log_prob + modinst.getWrapped(
                    meth, args_method_name="call", return_dict=False
                )(*[local_quantities[req] for req in modinst.requires])

            logger.descriptiveStatisticsTable(**local_quantities)
            return log_prob  # type: ignore

        return debug_log_prob

    def getTimedLogProb(self) -> Callable:

        def timed_log_prob(
            *state: Jnp_ndarray, compile: bool = True
        ) -> Union[Dict[str, float], Jnp_ndarray]:
            """

            Parameters
            ----------
            state: Jnp_ndarray :

            compile: bool :
                 (Default value = True)

            Returns
            -------


            """
            logger = Logger("Core:timed_log_prob")
            local_quantities = {}
            for varinst, varstate in zip(self._varinstances_sorted.values(), state):
                local_quantities.update(varinst.wrapStateToDict(varstate))

            times = {}
            for modinst in self.getInstances(self._inner_layers_sorted):
                logger.info(
                    f"Getting {modinst.provides} from {modinst.__class__.__name__}"
                )
                beg = time.time()
                local_quantities[modinst.provides] = jit(modinst, jit_compile=compile)(
                    *[local_quantities[req] for req in modinst.requires]
                )
                end = time.time()
                times[modinst.__class__.__name__] = end - beg

            log_prob = 0.0
            for modinst in self.getInstances(self._last_layer):
                logger.info(
                    f"Getting {modinst.provides} from {modinst.__class__.__name__}"
                )
                beg = time.time()
                log_prob = log_prob + jit(modinst, jit_compile=compile)(
                    *[local_quantities[req] for req in modinst.requires]
                )
                end = time.time()
                times[modinst.__class__.__name__] = end - beg

            return log_prob, times

        return timed_log_prob

    def checkGradientsLogProb(self, *state):
        """
        Use jax's check_grads to check the gradient of the `log_prob` function.

        There are problems of scaling for the tolerance, so it always fails.
        Can be useful if nan appears.

        Parameters
        ----------
        states : states to evaluate on


        Returns
        -------


        """
        self._logger.info("Checking gradients numerically...")
        self._logger.flush()
        try:
            check_grads(self.getLogProb(), *state, order=1)
            self._logger.success("Gradients have been tested numerically!")
        except AssertionError as e:
            self._logger.error(e)

    # RUN

    def generateInitialState(self, **kwargs: Any) -> Dict[str, Any]:
        """

        Parameters
        ----------
        verbosity: int :
             (Default value = 2)
        **kwargs: Any :


        Returns
        -------


        """
        logger = Logger("Core:generateInitialState", self._verbosity)

        seed = getKeysFromDicts(
            "seed",
            kwargs=kwargs,
            logger=logger,
        )

        if seed is None:
            seed = np.random.randint(2**32 - len(self._variables_layer_sorted))

        logger.assertIsInstance(seed, "seed", int)

        state = [
            jnp.atleast_1d(inst.generateInitialState(seed=seed + i))  # type: ignore
            for i, inst in enumerate(self._varinstances_sorted.values())  # type: ignore
        ]

        return {"current_state": state}

    def maximizeLogProb(self, **kwargs: Any) -> Dict[str, Any]:
        """ """
        logger = Logger("Core:maximizeLogProb", self._verbosity)
        (
            run,
            solver,
            solver_kwargs,
            variables_strategy,
            keep_failed,
            return_results,
            n_walkers,
        ) = getKeysFromDicts(
            "run",
            "solver",
            "solver_kwargs",
            "variables_strategy",
            "keep_failed",
            "return_results",
            "n_walkers",
            kwargs=kwargs,
            logger=logger,
        )

        if not run:
            return {}

        # Checking everything is okay
        self._logger.assertIsInstance(
            variables_strategy, "variables_strategy", Dict[str, Any]
        )
        self._logger.assertIsInstance(solver, "solver", str)
        self._logger.assertIsInstance(
            solver_kwargs, "solver_kwargs", Optional[Dict[str, Any]]
        )

        fixed_var_val, random_var_gen, opt_var_gen = {}, {}, {}
        for varname, varinst in self._varinstances_sorted.items():  # type: ignore
            self._logger.assertIsIn(
                varname, varname, variables_strategy, "variables_strategy"  # type: ignore
            )
            varstrat = self._getKeysFromDicts(varname, variables_strategy=variables_strategy)  # type: ignore
            strategy, fixed_value = self._getKeysFromDicts(
                "strategy",
                "fixed_value",
                **{f"variables_strategy[{varname}]": varstrat},
                error_not_found=[True, False],
            )
            self._logger.assertIsInstance(
                strategy, f"variables_strategy[{varname}][strategy]", str
            )
            self._logger.assertIsIn(
                strategy,
                f"variables_strategy[{varname}][strategy]",
                ["maximize", "randomize", "fix"],
                "accepted_values",
            )

            if strategy == "maximize":
                opt_var_gen[varname] = varinst.generateInitialState  # type: ignore
            elif strategy == "randomize":
                random_var_gen[varname] = varinst.generateInitialState  # type: ignore
            elif strategy == "fix":

                self._logger.assertIsNotNone(
                    fixed_value, f"variables_strategy[{varname}][fixed_value]"
                )

                if isInstance(
                    fixed_value,
                    f"initial_state[{varname}]",
                    Union[int, float, List[Union[int, float]]],
                )[0]:
                    fixed_var_val[varname] = f_cast(fixed_value)
                elif isinstance(fixed_value, str):
                    if fixed_value == "mean":
                        self._logger.Assert(
                            hasattr(varinst, "prior_mean"),
                            AttributeError(
                                f"{varname} does not have `prior_mean` attribute"
                            ),
                        )
                        fixed_var_val[varname] = f_cast(varinst.prior_mean)

                    elif fixed_value[:-4] != ".npy":
                        self._logger.error(
                            ValueError(
                                f"Can only import npy-format initial states. Got {fixed_value}"
                            )
                        )
                        try:
                            fixed_var_val[varname] = f_cast(
                                np.load(fixed_value, allow_pickle=False)
                            )
                        except Exception as e:
                            self._logger.error(
                                ValueError(
                                    f"Could not load initial_state_generator[{varname}] | "
                                    + "|".join(e.args)
                                )
                            )

        self._logger.Assert(
            len(opt_var_gen) > 0, ValueError("There is no value maximize over!")
        )

        self._logger.debug(
            f"Variables to maximize over: {printDict(opt_var_gen, embedded=True)}"
        )
        self._logger.debug(
            f"Randomized variables: {printDict(random_var_gen, embedded=True)}"
        )
        self._logger.debug(
            f"Fixed variables: {printDict(fixed_var_val, embedded=True)}"
        )

        # Running optimization
        log_prob = self.getLogProb()

        def gen_random_state():
            fixed_values = fixed_var_val.copy()
            init_state = []
            for varname in self._varinstances_sorted:
                if varname in random_var_gen:
                    fixed_values[varname] = random_var_gen[varname](
                        np.random.randint(2**32)
                    )
                elif varname in opt_var_gen:
                    init_state.append(opt_var_gen[varname](np.random.randint(2**32)))
                elif varname not in fixed_values:
                    raise ValueError("WHAT THE F")

            return jnp.concatenate(init_state), fixed_values

        def opt_to_lp(ostate, fixed_values):
            lp_state, c = [], 0
            for i, (varname, varinst) in enumerate(self._varinstances_sorted.items()):  # type: ignore
                s = varinst.output_size
                if varname in opt_var_gen:
                    if varinst.bijector is not None:
                        lp_state.append(varinst.bijector.forward(ostate[c : c + s]))
                    else:
                        lp_state.append(ostate[c : c + s])
                    c = c + s
                else:
                    lp_state.append(fixed_values[varname])
            return lp_state

        def _minimize(ostate, fixed_values):
            return -1 * log_prob(*opt_to_lp(ostate, fixed_values))

        max_iterations = solver_kwargs.pop("max_iterations", 1_000)  # type: ignore
        diff_rtol = solver_kwargs.pop("diff_rtol", 1e-3)  # type: ignore
        diff_atol = solver_kwargs.pop("diff_atol", 1e-3)  # type: ignore
        diff_init_rtol = solver_kwargs.pop("diff_init_rtol", 0.0)  # type: ignore
        verbosity = solver_kwargs.pop("verbosity", False)  # type: ignore

        final_states, final_lps, successes = [], [], []
        for i_walker in range(n_walkers):  # type: ignore
            self._logger.info(f"Running maximization chain {i_walker}/{n_walkers}")
            init_ostate, loc_fixed_values = gen_random_state()
            # print("INIT OSTATE")
            # print(init_ostate)
            # print("LOC FIXED VALUES")
            # printDict(loc_fixed_values)
            # print("LP STATE")
            # lp_state = opt_to_lp(init_ostate, loc_fixed_values)
            # print([st.shape for st in lp_state])

            _minimize(init_ostate, loc_fixed_values)

            final_ostate, lp, success = optax_minimize(
                lambda x: _minimize(x, loc_fixed_values),
                init_ostate,
                max_iterations,
                solver,
                diff_rtol,
                diff_atol,
                diff_init_rtol,
                print_f=self._logger.info,
                verbosity=verbosity,
                fdtype=fdtype,
                **solver_kwargs,  # type: ignore
            )
            final_states.append(opt_to_lp(final_ostate, loc_fixed_values))
            final_lps.append(lp)
            successes.append(success)

        final_lps = np.array(final_lps)

        w_succ = np.where(successes)[0]
        n_succ = w_succ.size

        if not keep_failed:
            final_states = [
                final_states[i] for i in range(len(final_states)) if i in w_succ
            ]
            final_lps = final_lps[w_succ, ...]

        to_print = (
            f"{n_succ} / {n_walkers} ({n_succ / n_walkers:.2%}) succeded\n"  # type: ignore
            f"lob_prob = {np.mean(final_lps):.2f} +- {np.std(final_lps):.2f}\n"
        )
        self._logger.info(to_print)

        if return_results == "best":
            best = np.argmin(final_lps)
            self._logger.info(f"Best run is {best}th")
            ret = {
                "current_state": final_states[best],
                "maximize:chain": [
                    jnp.stack([jnp.atleast_1d(st[i]) for st in final_states])
                    for i in range(len(final_states[0]))
                ],
            }
        else:
            raise NotImplementedError("TODO")

        return ret

    def runDebug(self, **kwargs: Any) -> Dict[str, Any]:
        (run, debug_classes, current_state) = self._getKeysFromDicts(
            "run", "debug_classes", "current_state", kwargs=kwargs, error_not_found=True
        )

        if not run:
            return {}

        self.getDebugLogProb()(*current_state, debug_classes=debug_classes)

        if run == "only":
            raise KeyboardInterrupt(
                "This is not an issue, just a shortcut: run == 'only' dies after running debug!"
            )
        return {}

    def runKernel(self, **kwargs: Any) -> Dict[str, Any]:
        """

        Parameters
        ----------
        verbosity: int :
             (Default value = 2)
        **kwargs: Any :


        Returns
        -------


        """
        logger = Logger("Core:runKernel", self._verbosity)

        # Getting args for the run
        (
            name_run,
            kernel,
            jit_compile,
            current_state,
            last_kernel_results,
            sample_chain_kwargs,
            time_per_step,
        ) = getKeysFromDicts(
            "name_run",
            "kernel",
            "jit_compile",
            "current_state",
            "last_kernel_results",
            "sample_chain_kwargs",
            "time_per_step",
            kwargs=kwargs,
            logger=logger,
            error_not_found=[True] * 6 + [False],
        )

        # Init kernel
        self._logger.assertIsNotNone(kernel, f"runKernel:{name_run}:kernel")
        kerclass = self.getClasses(kernel)
        ker_def_kwargs = [k for k in dir(kerclass) if "kwargs" in k]
        kernel_kwargs = {k: v for k, v in kwargs.items() if k in ker_def_kwargs}
        logger.debug(f"Kernel expects kwargs {ker_def_kwargs}")
        logger.debug(f"Kernel kwargs are \n{printDict(kernel_kwargs, embedded=True)}")
        kernel_instance = kerclass(
            varinstances_sorted=self._varinstances_sorted.values(),
            log_prob_fn=self.getLogProb(),
            verbosity=self._verbosity,
            **kernel_kwargs,
        )

        # Checking for the sample_chain_kwargs
        logger.assertIsIn(
            *[
                [
                    "num_results",
                    "num_burnin_steps",
                    "num_steps_between_results",
                    "return_final_kernel_results",
                    "seed",
                ]
            ]
            * 2,
            sample_chain_kwargs,
            "sample_chain_kwargs",
        )

        # Setting up seed jax-style
        sample_chain_kwargs["seed"] = getJaxRandomKey(sample_chain_kwargs["seed"])

        logger.assertIsNotIn(
            None, "None", sample_chain_kwargs.values(), "sample_chain_kwargs"
        )

        logger.info(
            f"Running {kernel} with kwargs\n{printDict(kernel_kwargs, embedded=True)}\n"
            f"and sample kwargs\n{printDict(sample_chain_kwargs, embedded=True)}"
        )

        if last_kernel_results is not None:
            logger.debug(
                "Build previous_kernel_results from provided last_kernel_results"
            )
            type_kernel_results_expected = type(
                kernel_instance.getKernel().bootstrap_results(current_state)
            )
            while True:
                self._logger.debug(
                    f"At this stage, last_kernel_results is \n{last_kernel_results}"
                )
                if isinstance(last_kernel_results, type_kernel_results_expected):
                    break
                try:
                    last_kernel_results = getattr(last_kernel_results, "inner_results")
                except AttributeError as error:
                    error.args = error.args + (
                        "Could not find inner results of type "
                        f"{type_kernel_results_expected} in {last_kernel_results}",
                    )
                    self._logger.error(error)
            self._logger.debug(
                f"Found last_kernel_results of type {type_kernel_results_expected}"
            )

        def maybeCompiledRun():
            return tfp.mcmc.sample_chain(
                kernel=kernel_instance.getKernel(),
                trace_fn=kernel_instance.trace,
                current_state=current_state,
                previous_kernel_results=last_kernel_results,
                **sample_chain_kwargs,
            )

        if jit_compile:
            maybeCompiledRun = jit(maybeCompiledRun)

        n_steps = (
            sample_chain_kwargs["num_results"] + sample_chain_kwargs["num_burnin_steps"]
        ) * max([1, sample_chain_kwargs["num_steps_between_results"]])
        logger.debug(f"{n_steps} steps to run")
        chain_size_MB = sample_chain_kwargs["num_results"] * self._state_size_MB
        logger.info(f"Memory used by chain (alone): {chain_size_MB} MB")
        logger.info(f"Starting kernel run {name_run} at {time.ctime()}")
        if time_per_step is not None:
            logger.info(f"Expected running time: {time2hms(time_per_step * n_steps)}")
            logger.info(
                f"Expected end time: {time.ctime(time.time() + time_per_step * n_steps)}"
            )

        logger.flush()
        t0 = time.time()
        chain, trace, lkr = maybeCompiledRun()
        t1 = time.time()

        logger.success(
            f"Done running {name_run} at {time.ctime()} in {time2hms(t1 - t0)}"
        )
        time_per_step = (t1 - t0) / n_steps

        trace = [np.array(tr) for tr in trace]
        return {
            "last_kernel_results": lkr,
            "current_state": [chain[-1, :] for chain in chain],
            f"{name_run}:chain": [atLeast2DLastAxis(varchain) for varchain in chain],
            f"{name_run}:trace": trace,
            f"{name_run}:kernel_instance": kernel_instance,
            f"{name_run}:time_kernel_run": t1 - t0,
            f"{name_run}:n_steps": n_steps,
            "time_per_step": time_per_step,
        }

    # def runKernel(self, **kwargs: Any) -> Dict[str, Any]:
    #     """

    #     Parameters
    #     ----------
    #     verbosity: int :
    #          (Default value = 2)
    #     **kwargs: Any :

    #     Returns
    #     -------

    #     """
    #     logger = Logger("Core:runKernel", self._verbosity)

    #     # Getting args for the run
    #     (
    #         name_run,
    #         current_state,
    #         last_kernel_results,
    #         nuts_bj_kwargs,
    #         time_per_step,
    #     ) = getKeysFromDicts(
    #         "name_run",
    #         "current_state",
    #         "last_kernel_results",
    #         "nuts_bj_kwargs",
    #         "time_per_step",
    #         kwargs=kwargs,
    #         logger=logger,
    #         error_not_found=[True, False, False, True, False],
    #     )

    #     (
    #         n_results,
    #         n_adapt,
    #         seed,
    #         n_steps_between_results,
    #         step_size,
    #         nuts_kwargs,
    #         da_kwargs,
    #     ) = getKeysFromDicts(
    #         "n_results",
    #         "n_adapt",
    #         "seed",
    #         "n_steps_between_results",
    #         "step_size",
    #         "nuts_kwargs",
    #         "da_kwargs",
    #         kwargs=nuts_bj_kwargs,  # type: ignore
    #         logger=logger,
    #     )

    #     n_steps = n_results * n_steps_between_results  # type: ignore
    #     logger.debug(f"{n_steps} steps to run")
    #     chain_size_MB = n_results * self._state_size_MB
    #     logger.info(f"Memory used by chain (alone): {chain_size_MB} MB")
    #     logger.info(f"Starting kernel run {name_run} at {time.ctime()}")
    #     if time_per_step is not None:
    #         logger.info(f"Expected running time: {time2hms(time_per_step * n_steps)}")
    #         logger.info(
    #             f"Expected end time: {time.ctime(time.time() + time_per_step * n_steps)}"
    #         )

    #     logger.flush()
    #     t0 = time.time()
    #     chain, trace, lkr = run_nuts(
    #         self.getLogProb(), current_state, safeDivide(jnp.concatenate)
    #     )
    #     t1 = time.time()

    #     logger.success(
    #         f"Done running {name_run} at {time.ctime()} in {time2hms(t1 - t0)}"
    #     )
    #     time_per_step = (t1 - t0) / n_steps

    #     trace = [np.array(tr) for tr in trace]
    #     return {
    #         "last_kernel_results": lkr,
    #         "current_state": [chain[-1, :] for chain in chain],
    #         f"{name_run}:chain": [atLeast2DLastAxis(varchain) for varchain in chain],
    #         f"{name_run}:trace": trace,
    #         f"{name_run}:kernel_instance": kernel_instance,
    #         f"{name_run}:time_kernel_run": t1 - t0,
    #         f"{name_run}:n_steps": n_steps,
    #         "time_per_step": time_per_step,
    #     }

    def summarize(self, **kwargs: Any) -> Dict[str, Any]:
        """

        Parameters
        ----------
        verbosity: int :
             (Default value = 2)
        **kwargs: Any :


        Returns
        -------


        """
        logger = Logger("Core:summarize", self._verbosity)
        name_run = getKeysFromDicts("name_run", kwargs=kwargs, logger=logger)
        (
            chain,
            trace,
            kernel_instance,
            time_kernel_run,
            n_steps,
            time_per_step,
        ) = getKeysFromDicts(
            f"{name_run}:chain",
            f"{name_run}:trace",
            f"{name_run}:kernel_instance",
            f"{name_run}:time_kernel_run",
            f"{name_run}:n_steps",
            "time_per_step",
            kwargs=kwargs,
            logger=logger,
        )

        to_print = f"Run time is {time2hms(time_kernel_run)} for {n_steps} steps, "
        to_print += f"i.e. {time2hms(time_per_step)} per step\n"
        kernel_instance.summarize(trace, name_run)  # type: ignore

        for varinst, varchain in zip(self._varinstances_sorted.values(), chain):  # type: ignore
            varinst.summarize(varchain, name_run)  # type: ignore

        return {}

    def saveResults(self, **kwargs: Any) -> Dict[str, Any]:
        """

        Parameters
        ----------
        verbosity: int :
             (Default value = 2)
        **kwargs: Any :


        Returns
        -------


        """
        logger = Logger("Core:saveResults", self._verbosity)
        command = getKeysFromDicts("command", kwargs=kwargs, logger=logger)  # type: ignore

        logger.assertIsIn(
            command,
            command,  # type: ignore
            ["generateInitialState", "maximizeLogProb", "runKernel"],
            "available commands",
        )
        if command == "generateInitialState":
            current_state = getKeysFromDicts(  # type: ignore
                "current_state", kwargs=kwargs, logger=logger
            )
            self._saveStateLike(current_state, "current_state")  # type: ignore
        elif command == "maximizeLogProb":
            current_state = getKeysFromDicts(
                "current_state", kwargs=kwargs, logger=logger
            )
            self._saveStateLike(current_state, "current_state")  # type: ignore
        elif command == "runKernel":
            name_run = getKeysFromDicts("name_run", kwargs=kwargs, logger=logger)
            (
                current_state,
                last_kernel_results,
                chain,
                trace,
                kernel_instance,
                save_mode,
            ) = getKeysFromDicts(
                "current_state",
                "last_kernel_results",
                f"{name_run}:chain",
                f"{name_run}:trace",
                f"{name_run}:kernel_instance",
                "save_mode",
                kwargs=kwargs,
                logger=logger,
            )
            # self._saveOtherRunOutputs()
            self._saveStateLike(current_state, "current_state")  # type: ignore
            self._saveLastKernelResults(last_kernel_results, name_run)  # type: ignore
            self._saveChain(chain, name_run, save_mode)  # type: ignore
            self._saveTrace(trace, name_run, kernel_instance.trace_return, save_mode)  # type: ignore

        return {}

    def runTraceAnalysis(self, **kwargs: Any) -> Dict[str, Any]:
        logger = Logger("Core:analyze", self._verbosity)
        name_run = getKeysFromDicts("name_run", kwargs=kwargs, logger=logger)
        (
            trace,
            kernel_instance,
            nr_kernel,
            kernel,
            time_kernel_run,
        ) = getKeysFromDicts(
            f"{name_run}:trace",
            f"{name_run}:kernel_instance",
            f"{name_run}:kernel",
            "kernel",
            f"{name_run}:time_kernel_run",
            kwargs=kwargs,
            logger=logger,
            error_not_found=[True] + [False] * 4,
        )

        if trace is not None:
            # if we have the instance, great, if not, we can do with the class
            if kernel_instance is not None:
                ker_analyze = kernel_instance.analyze  # type: ignore
            elif kernel is not None:
                ker_analyze = self.getClasses(kernel).analyze  # type: ignore
            elif nr_kernel is not None:
                ker_analyze = self.getClasses(nr_kernel).analyze  # type: ignore
            else:
                self._logger.info("Could not build kernel, trace will not be analyzed")
                ker_analyze = None

            if ker_analyze is not None:
                logger.info("Analysing trace")
                ker_analyze(trace, time_kernel_run, name_run, output_dir=self._io.anal_dir)  # type: ignore

        return {}

    def runAnalysis(self, **kwargs: Any) -> Dict[str, Any]:
        """

        Parameters
        ----------
        verbosity: int :
             (Default value = 2)
        **kwargs: Any :


        Returns
        -------


        """
        logger = Logger("Core:analyze", self._verbosity)
        name_run = getKeysFromDicts("name_run", kwargs=kwargs, logger=logger)
        (
            chain,
            save_level,
            plot_level,
            subsample,
        ) = getKeysFromDicts(
            f"{name_run}:chain",
            "save_level",
            "plot_level",
            "subsample",
            kwargs=kwargs,
            logger=logger,
            error_not_found=[True] + [False] * 3,
        )

        analyses = self.getInstances(
            self._selected_analnames_initattr, return_type="dict"  # type: ignore
        )
        keep_quantities_one_state, keep_quantities_finalize = [], []
        analnames_one_state_loc2glob = {}
        analnames_finalize_loc2glob = {}
        for name, inst in analyses.items():  # type: ignore
            locals, globals, *_ = inst.getArgs(
                "oneState", self._verbosity, ignore=["self", "real"]
            )
            keep_quantities_one_state += globals
            analnames_one_state_loc2glob[name] = [locals, globals]

            locals, globals, *_ = inst.getArgs(
                "finalize", self._verbosity, ignore=["self"]
            )
            keep_quantities_finalize += globals
            analnames_finalize_loc2glob[name] = [locals, globals]

            inst.prepare(
                plotting_output_dir=self._io.anal_dir,  # type: ignore
                results_output_dir=self._io.res_dir,  # type: ignore
                save_level=save_level,
                plot_level=plot_level,
            )

        _all_provided = []
        for varinst in self._varinstances_sorted.values():  # type: ignore
            _all_provided += varinst.provides

        keep_quantities_one_state = list(np.unique(keep_quantities_one_state))
        keep_quantities_finalize = list(
            np.unique(
                [
                    quant
                    for quant in keep_quantities_finalize
                    if quant not in _all_provided  # type: ignore
                ]
            )
        )

        keep_quantities = list(
            np.unique(keep_quantities_one_state + keep_quantities_finalize)
        )

        detailsLogProb = jit(self.getDetailsLogProb(keep_quantities))

        kept_for_finalize = {}

        for varchain, varinst in zip(chain, self._varinstances_sorted.values()):  # type: ignore
            for svchain, prov in zip(varchain.T, varinst.provides_free):  # type: ignore
                kept_for_finalize[prov] = atLeast2DLastAxis(svchain)
            for prov, val in zip(varinst.provides_fixed, varinst.provides_fixed_values):
                kept_for_finalize[prov] = val

        # print("DEBUG")
        # print(len(chain))
        # print([ch.shape for ch in chain])
        if subsample is not None:
            logger.assertIsInstance(subsample, "subsample", List[int])  # type: ignore
            iterator = list(range(*subsample))  # type: ignore
            logger.Assert(
                iterator[0] >= 0 and iterator[-1] < chain[0].shape[0],  # type: ignore
                ValueError(
                    f"Subsample {iterator} not in [0, {chain[0].shape[0]}]"
                ),  # type: ignore,
            )
        else:
            iterator = range(chain[0].shape[0])  # type: ignore

        n_ite = len(iterator)

        if self._verbosity <= 1:
            iterator = tqdm(iterator, desc="Analysing states")  # type: ignore

        for ireal, real in enumerate(iterator):
            state = [
                np.reshape(
                    varchain,
                    (varchain.shape[0], varchain.shape[1] if varchain.ndim == 2 else 1),
                )[real, :]
                for varchain in chain
            ]  # type: ignore
            t0 = time.time()
            details = detailsLogProb(*state)
            t1 = time.time()
            logger.debug(f"Ran details log prob in {t1 - t0}s")

            for name, inst in analyses.items():  # type: ignore
                logger.debug(f"Running {name}.oneState (real={real})")
                locals, globals = analnames_one_state_loc2glob[name]
                t0 = time.time()
                inst.oneState(
                    real, **{loc: details[glob] for loc, glob in zip(locals, globals)}
                )
                t1 = time.time()
                logger.debug(f"Ran {name}.oneState in {t1 - t0}s")

            for q, v in details.items():
                if q not in keep_quantities_finalize:
                    continue

                v = np.squeeze(v)  # 1d to float
                if q in kept_for_finalize:
                    kept_for_finalize[q] = np.concatenate(
                        [kept_for_finalize[q], [v]], axis=0
                    )
                else:
                    kept_for_finalize[q] = np.stack([v])

            if real > 0:
                t2 = time.time()
                logger.debug(time2hms((t2 - t0) / ireal * n_ite) + " left")

        for name, inst in analyses.items():  # type: ignore
            logger.debug(f"Running {name}.finalize")
            locals, globals = analnames_finalize_loc2glob[name]
            inst.finalize(
                **{loc: kept_for_finalize[glob] for loc, glob in zip(locals, globals)}
            )

        return {}

    def loadForRerun(self, **kwargs) -> Dict[str, Any]:
        """

        Parameters
        ----------
        verbosity: int :
             (Default value = 2)
        **kwargs :


        Returns
        -------


        """
        logger = Logger("loadForRerun", self._verbosity)
        name_run = getKeysFromDicts("name_run", kwargs=kwargs, logger=logger)

        lkr = self._io.loadDill("res", f"{name_run}_last_kernel_results")
        cs = self._loadStateLike("current_state")
        return {"last_kernel_results": lkr, "current_state": cs}

    def loadResults(self, verbosity: int = 2, **kwargs) -> Dict[str, Any]:
        """

        Parameters
        ----------
        verbosity: int :
             (Default value = 2)
        **kwargs :


        Returns
        -------


        """
        logger = Logger("loadResults", verbosity)
        name_run, kernel = getKeysFromDicts(
            "name_run", "kernel", kwargs=kwargs, logger=logger
        )

        trace_return = self.getClasses(kernel).trace_return  # type: ignore

        trace = [
            self._io.loadNpy("res", f"{name_run}_{tr}", error_not_found=False)
            for tr in trace_return
        ]

        trace = None if any(tr is None for tr in trace) else np.stack(trace)

        return {
            f"{name_run}:chain": [
                atLeast2DLastAxis(
                    self._io.loadNpy(
                        "res", f"{name_run}_{'_'.join(varinst.provides_free)}"
                    )
                )
                for varinst in self._varinstances_sorted.values()  # type: ignore
            ],
            f"{name_run}:trace": trace,
        }

    def forgetChain(self, verbosity: int = 2, **kwargs):
        logger = Logger("forgetChain", verbosity)
        name_run = getKeysFromDicts("name_run", kwargs=kwargs, logger=logger)
        kwargs.pop("name_run")  # type: ignore
        logger.info(f"Forgetting everything about {name_run}: {list(kwargs.keys())}")

        for k, v in kwargs.items():
            size = v.__sizeof__() / 1024**2
            logger.debug(f"Deletting {k} of size {size:.0f}Mb")
            del v

        return {k: None for k in kwargs}

    def runStrategy(self) -> Dict[str, Any]:
        """

        Parameters
        ----------
        verbosity: int :
             (Default value = 2)
        save_core_state: bool :
             (Default value = True)

        Returns
        -------


        """

        logger = Logger("Core:runStrategy", verbosity=self._verbosity)

        self._logger.debug(
            f"strategy init kwargs\n{printDict(self._strategy_init_kwargs, embedded=True)}"
        )
        strategy_instance = self._getKeysFromDicts(
            self._strategy, all_loaded_classname=self._all_loaded_classnames_classes
        )(verbosity=self._verbosity, **self._strategy_init_kwargs)

        updated_cmd_kwargs = {}
        for command, default_cmd_kwargs in strategy_instance:
            logger.info(f"Running command {command}")
            logger.debug(
                f"General common kwargs \n{printDict(updated_cmd_kwargs, embedded=True)}"
            )
            logger.debug(
                f"Default command kwargs \n{printDict(default_cmd_kwargs, embedded=True)}"
            )
            func = getattr(self, command)

            kwargs = updateDictRecursive(
                default_cmd_kwargs, updated_cmd_kwargs, logger=logger
            )
            logger.debug(f"Kwargs given to func \n{printDict(kwargs, embedded=True)}")
            new_kwargs = func(**kwargs)
            logger.debug(
                f"Kwargs received from func \n{printDict(new_kwargs, embedded=True)}"
            )
            updated_cmd_kwargs.update(new_kwargs)
            logger.success(f"Succesfully ran command {command}")
        return updated_cmd_kwargs

    # DEBUG

    def _plotHistogramDebug(self, value, target, name):
        """

        Parameters
        ----------
        value :

        target :

        name :


        Returns
        -------


        """
        fig, ax = plt.subplots(1)
        mmin, mmax = np.min([value, target]), np.max([value, target])
        bins = np.linspace(mmin, mmax, 50)
        ax.hist(value, bins, histtype="step", color="b", label="value")
        ax.hist(target, bins, histtype="step", color="k", label="target")
        ax.hist(value - target, bins, histtype="step", color="r", label="error")
        ax.legend()
        ax.grid()
        ax.set_xlabel(name)
        fig.savefig(f"{self._io.debug_dir}/test_details_log_prob_{name}.pdf")

    def testDetailsLogProb(
        self, target_details: Dict[str, np.ndarray], plot_hists: bool = True
    ) -> None:
        """

        Parameters
        ----------
        target_details: Dict[str :

        np.ndarray] :

        plot_hists: bool :
             (Default value = True)

        Returns
        -------


        """
        logger = Logger("Core:testDetailsLogProb")
        state = self.concatenateState(target_details)
        details = self.details_log_prob(state)

        to_print = ""
        for key, target in target_details.items():
            if key in details:
                val = np.array(details[key])
                to_print += f"### {key} ###\n"
                to_print += f"target = {target}\n"
                to_print += f"value  = {val}\n"
                to_print += f"error  = {val - target}\n"
                if plot_hists and len(val) > 10:
                    logger.debug(f"Plotting histogram for {key}")
                    self._plotHistogramDebug(val, target, key)
            if "stats" in key:
                key = key.replace(":stats", "")
                val = np.array(details[key])
                to_print += f"### {key} ###\n"
                to_print += descriptiveStatistics(val, reference=target)[0] + "\n"

        logger.info(f"Results: \n{to_print}")

    def testRunTime(
        self, state: Union[np.ndarray, Jnp_ndarray], n_test: int = 100, n_eval: int = 1
    ):
        """

        Parameters
        ----------
        state: Union[np.ndarray :

        Jnp_ndarray] :

        n_test: int :
             (Default value = 100)
        n_eval: int :
             (Default value = 1)

        Returns
        -------


        """
        logger = Logger("Core:testRunTime")
        logger.info("Compiling modules one by one...", flush=True)
        _, times = self.timed_log_prob(state)

        # timing modules
        logger.info("Timing modules...", flush=True)
        _, times = self.timed_log_prob(state)
        tot = np.sum(list(times.values()))
        to_print = f"{'Name':50s} {'Time (s)':20>s} {'Percent':20>s}\n"
        for modname, mtime in times.items():
            to_print += f"{modname:50s} {mtime:20f} {100 * mtime/tot:20.0f} \n"
        to_print += f"{'total':50s} {tot:20f}"
        logger.info("Modules' run times:\n" + to_print, flush=True)

        # timing log_prob
        logger.info("Compiling log_prob...", flush=True)

        @jit(jit_compile=True)
        def _run(state):
            """

            Parameters
            ----------
            state :


            Returns
            -------


            """
            return self.log_prob(state)

        _run(state)

        rtimes = []
        logger.info(f"Running {n_test} times the (compiled) log_prob...", flush=True)
        for _ in range(n_test):
            beg = time.time()
            _run(state)
            end = time.time()
            rtimes.append(end - beg)
        # print()
        mrtime, srtime, minrtime, maxrtime = (
            np.mean(rtimes),
            np.std(rtimes),
            np.min(rtimes),
            np.max(rtimes),
        )
        self._logger.info(
            f"Ran {n_test} times the (compiled) log_prob, "
            f"run time is {minrtime} < {mrtime} +- {srtime} < {maxrtime}",
            flush=True,
        )
        self._logger.info(
            f"Expected run time for {n_eval} evalutions: {mrtime * n_eval} "
            f"+- {srtime / np.sqrt(n_eval)} /!\ not taking gradients into account",
            flush=True,
        )

    # SAVE

    def _saveStateLike(
        self,
        data: List[Jnp_ndarray],
        name: str,
    ) -> None:
        """

        Parameters
        ----------
        data: Jnp_ndarray :

        name: str :


        Returns
        -------


        """
        for i, varinst in enumerate(self._varinstances_sorted.values()):
            self._io.saveNpy("res", f"{name}_{varinst.provides}", data[i])

    def _loadStateLike(
        self,
        name: str,
    ) -> List[np.ndarray]:
        return [
            self._io.loadNpy("res", f"{name}_{varinst.provides}")
            for varinst in self._varinstances_sorted.values()
        ]

    def _saveLastKernelResults(
        self,
        data: Any,
        name_run: str,
    ) -> None:
        """

        Parameters
        ----------
        data: Any :

        name_run: str :


        Returns
        -------


        """
        self._io.saveDill("res", f"{name_run}_last_kernel_results", data)

    def _saveChain(
        self,
        chain: List[Jnp_ndarray],
        name_run: str,
        save_mode: str,
    ) -> None:
        """

        Parameters
        ----------
        data: List[List[Jnp_ndarray]] :

        name_run: str :


        Returns
        -------


        """
        for varinst, varchain in zip(self._varinstances_sorted.values(), chain):
            self._io.saveNpy(
                "res",
                f"{name_run}_{'_'.join(varinst.provides_free)}",
                varchain,  # type: ignore
                save_mode,  # type: ignore
            )

    def _saveTrace(
        self,
        traces: List[Jnp_ndarray],
        name_run: str,
        trace_return: List[str],
        save_mode: str,
    ) -> None:
        """

        Parameters
        ----------
        data: List[Jnp_ndarray] :

        name_run: str :

        trace_return: List[str] :


        Returns
        -------


        """
        for trace_ret, trace_val in zip(trace_return, traces):
            self._io.saveNpy("res", f"{name_run}_{trace_ret}", trace_val, save_mode)

    # BEAUTIFUL PROMPTING OF AVAILABLE CLASSES

    # base prompting

    def _headerBase(self):
        """ """
        header = f" {'Name':<50} | Library"
        header += "\n" + "-" * len(header)
        return header

    def _oneLineBase(self, name):
        """

        Parameters
        ----------
        name :


        Returns
        -------


        """
        return f" {name:<50} | " f"{self._all_loaded_classnames_libfiles[name]}"

    # list modules

    def _headerMod(self):
        """ """
        header = f" {'Name':<40} | {'Requires':<50} | {'Provides':<30} | Library"
        header += "\n" + "-" * len(header)
        return header

    def _oneLineMod(self, name):
        """

        Parameters
        ----------
        name :


        Returns
        -------


        """
        modclass = self._all_loaded_classnames_classes[name]
        req = ", ".join(modclass.requires)
        pro = ", ".join(modclass.provides)
        return (
            f" {name:<40} | {req:<50} | {pro:<30} | "
            f"{self._all_loaded_classnames_libfiles[name]}"
        )

    # list variables

    def _headerVar(self):
        """ """
        header = f" {'Name':<40} | {'Provides':<50} | Library"
        header += "\n" + "-" * len(header)
        return header

    def _oneLineVar(self, name):
        """

        Parameters
        ----------
        name :


        Returns
        -------


        """
        varclass = self._all_loaded_classnames_classes[name]
        pro = ", ".join(varclass.provides)
        return (
            f" {name:<40} | "
            f"{pro:<50} | "
            f"{self._all_loaded_classnames_libfiles[name]}"
        )

    # The listing function to call

    def listLoadedClasses(self, typeclass="mod", select=None, embedded=False):
        """
        Beautiful listing of the available classes of type `type`

        Parameters
        ----------
        typeclass: str:
            Values in mod, var, util, anal, ker, strat
        select :
             (Default value = None)
        embedded :
             (Default value = False)

        Returns
        -------


        """
        all_loaded = getattr(self, f"_all_loaded_{typeclass}names")
        try:
            header_func = getattr(self, f"_header{typeclass.capitalize()}")
            oneline_func = getattr(self, f"_oneLine{typeclass.capitalize()}")
        except AttributeError:
            header_func = getattr(self, f"_headerBase")
            oneline_func = getattr(self, f"_oneLineBase")

        if select is None:
            names = all_loaded
        else:
            names = [name for name in all_loaded if name in select]

        string = (
            "\n"
            + header_func()
            + "\n"
            + "\n".join([oneline_func(name) for name in names])
            + "\n"
        )
        if not embedded:
            print(string)
        else:
            return string
