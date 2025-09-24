import sys
import os
from os.path import join, isfile, isdir
import shutil as shu
import yaml
import dill

from typing import Any, Dict, Optional, List, Union, Tuple

import numpy as np
import pandas as pd

from .logger import Logger
from .common import banana_path
from .utils import printDict, getKeysFromDicts, conditionalCastTo


def makedirs(path):
    """
    Very lazy shortcut
    """
    os.makedirs(path, exist_ok=True)


class ConfigIO:

    # Structure of the sections
    # parameters is pretty deep, we check later the inner structure
    # root is Dict[str, Any]
    # each section is a Dict[str, Any]
    _template_config = dict(
        general=dict(
            mandatory=True,
            content=dict(
                other=False,
                description=str,
                working_dir=str,
                libraries_dirs=dict(
                    _type=Optional[Union[List[str], str]],
                    cast=((None, []), (str, lambda x: [x])),
                ),
                load_working_dir=bool,
                copy_data=bool,
                copy_libraries=bool,
            ),
        ),
        logging=dict(
            mandatory=True,
            content=dict(
                other=False,
                verbosity=int,
                debug_classes=dict(
                    _type=Optional[Union[List[str], str]], cast=((None, "none"))
                ),
                highlights=dict(
                    _type=Optional[Union[List[str], str]],
                    cast=((None, []), (str, lambda x: [x])),
                ),
            ),
        ),
        strategy=dict(
            mandatory=True,
            content=dict(other=Dict[str, Any], strategy=str),
        ),
        parameters=dict(
            mandatory=False,
            content=dict(
                other=Optional[Union[bool, int, float, List, Dict]],
                # Recursive checking of the sub-dicts is done in the hook
            ),
        ),
        data=dict(
            mandatory=False,
            _type=Dict[str, Dict],  # inner-dict type type defined in content
            content=dict(
                other=Optional[
                    Dict[
                        str,
                        Union[
                            bool,
                            str,
                            # np.ndarray,
                            Dict[str, Any],
                            List[str],
                            List[int],
                            Tuple[int],
                        ],
                    ]
                ]
            ),
        ),
        variables=dict(
            mandatory=False,
            content=dict(other=Optional[Dict[str, Any]]),
        ),
        modules=dict(
            mandatory=False,
            content=dict(other=Optional[Dict[str, Any]]),
        ),
        analyses=dict(
            mandatory=False,
            content=dict(other=Optional[Dict[str, Any]]),
        ),
        utils=dict(
            mandatory=False,
            content=dict(other=Optional[Dict[str, Any]]),
        ),
    )

    # Map to core attributes

    _config_core_map = dict(
        general=dict(
            description=True,
            working_dir=True,
            libraries_dirs=True,
            copy_libraries=True,
        ),
        logging=dict(
            verbosity=True,
            debug_classes=True,
            highlights=False,
        ),
        strategy=dict(
            other="_strategy_init_kwargs",
            strategy=True,
        ),
        parameters=True,
        data=True,
        variables="_selected_varnames_initattr",
        modules="_selected_modnames_initattr",
        analyses="_selected_analnames_initattr",
        utils="_selected_utilnames_initattr",
    )

    def __init__(
        self,
        io,
        verbosity: int = 2,
    ):
        self._logger = Logger("Config", verbosity)
        self._io = io

    def getConfig(self) -> Dict[str, Any]:
        return self._config

    def _getKeysFromDicts(self, *args, **kwargs):
        return getKeysFromDicts(*args, logger=self._logger, **kwargs)

    # Load / Dump

    def _loadFile(self, config_file: str, argv: List[str] = []) -> Dict[str, Any]:
        self._logger.info(f"Loading configuration from {config_file}")

        with open(config_file, "r") as f:
            content = f.read()
        content = os.path.expandvars(content)

        for arg in argv:
            self._logger.Assert(
                "=" in arg,
                ValueError("Passed arguments should have the form KEY=VALUE"),
            )
            w = arg.find("=")
            k, v = arg[:w], arg[w + 1 :]
            self._logger.debug(f"Replacing {k} with {v} in config file")
            content = content.replace("${" + k + "}", v)

        if "${" in content:  # }
            self._logger.warn(
                "There seems to be some unresolved ${KEY} in the configuration file"
                "Are you sure you properly passed the `KEY=VALUE` args?"
            )

        config = yaml.safe_load(content)

        self._logger.debug(
            f"Loaded from {config_file}: \n{printDict(config, embedded=True)}"
        )

        return config

    def recursiveLoad(
        self,
        config: Union[str, Dict[str, Any]],
        argv: List[str] = [],
    ) -> None:
        """
        Loads the yaml config.
        Environment variables are replaced (before yaml interpretation)
        as well as passed argv of the form `KEY=VALUE`.

        In the yaml file, the keys to replace must take the form `${KEY}`.
        Warning: if a `${KEY}` is present in the yaml file, but not `KEY=VALUE` is passed,
        it is left as such in the file. A warning will be issued.

        argv are passed explicitely (vs gotten from sys.argv) to simplify the
        initialization from a python environment

        If load_working_dir, the config.yml of the working is loaded first and then updated with the
        current configuration.
        """

        # def _aux(config):

        #     try:
        #         base_config_file = config["general"]["base_config"]
        #     except KeyError:
        #         # ill formated but we'll see that later
        #         return config

        #     if isinstance(base_config_file, str):
        #         base_config = _aux(self._loadFile(base_config_file, argv))
        #         # we overwrite the base values
        #         base_config.update(config)
        #         return base_config
        #     else:
        #         return config

        if isinstance(config, str):
            config = self._loadFile(config, argv)
        elif not isinstance(config, dict):
            self._logger.error(
                TypeError("Config should be a str (path to yaml file) or a dict!")
            )

        try:
            lwd = config["general"]["load_working_dir"]
            working_dir = config["general"]["working_dir"]
        except KeyError:
            # Will crash later
            self._config = config
            return

        if lwd:
            # we keep both, for debug and check if data should have been copied
            self._base_config = self._loadFile(f"{working_dir}/config.yml", argv)
            self._config = self._base_config.copy()
            # We overwrite base config
            self._config.update(config)
        else:
            self._config = config

    def dump(self, config_file: str) -> None:
        """
        Saving to config_file in yaml format.

        We don't save the data in the config file
        for now, reloading a configuration is possible only with `copy_data = True`
        """

        _config = self._config.copy()
        _config["data"] = {k: None for k in _config["data"].keys()}

        with open(config_file, "w") as f:
            self._logger.info(f"Saving configuration in {config_file}")
            self._logger.debug(f"To save\n{printDict(self._config, embedded=True)}")
            yaml.dump(_config, f)

    # Check validity of the config

    def _checkValidityAndRecastContent(
        self, section: str, content_template: Dict[str, Any]
    ):
        """
        Checking the format of the configuration and recasting when necessary.
        Restricted to a single section.
        """
        sec_attributes = self._config[section]

        other_template = content_template.pop("other")
        for item, item_template in content_template.items():
            self._logger.debug(f"Checking and recasting {item}")
            if isinstance(item_template, dict):
                _type = item_template["_type"]
                cast = item_template["cast"]
            else:
                _type = item_template
                cast = None

            self._logger.assertIsIn(item, item, sec_attributes, f"{section}_section")
            self._logger.assertIsInstance(
                sec_attributes[item], f"{section}:{item}", _type
            )

            if cast is not None:
                # LHS: we use full "path" to avoid possible issues
                # with non modifiable types, even though all sections
                # should be dicts and lists
                self._config[section][item] = conditionalCastTo(
                    sec_attributes[item],
                    f"{section}:{item}",
                    cast,
                    logger=self._logger,
                    error_uncasted=False,
                )

        # working on the "other" items
        if isinstance(other_template, dict):
            _type = other_template["_type"]
            cast = other_template["cast"]
        else:
            _type = other_template
            cast = None

        for item, val in sec_attributes.items():
            self._logger.debug(f"Checking and recasting 'other' item: {item}")
            # already treated in the loop above
            if item in content_template:
                continue

            # we dont accept other args
            if other_template is False:
                self._logger.error(
                    ValueError(f"{section}:{item} is not a valid argument of {section}")
                )

            # check type and recast
            else:
                self._logger.assertIsInstance(val, f"{section}:{item}", _type)

                if cast is not None:
                    # LHS: we use full "path" to avoid possible issues
                    # with non modifiable types, even though all sections
                    # should be dicts and lists
                    self._config[section][item] = conditionalCastTo(
                        sec_attributes[item],
                        f"{section}:{item}",
                        cast,
                        logger=self._logger,
                        error_uncasted=False,
                    )

    def checkValidityAndRecast(self):
        """
        Checking the format of the configuration and recasting when necessary.
        This function parses self._template_config and the self._config and compares the types, etc.
        """
        for section, sec_template in self._template_config.items():
            self._logger.debug(f"Checking and recasting section {section}")
            mandatory = sec_template["mandatory"]
            if section not in self._config:
                if mandatory:
                    self._logger.error(ValueError(f"Section {section} is mandatory"))
                else:
                    self._logger.info(
                        f"Optional section {section} is not found in configuration"
                    )
                    continue

            sec_attributes = self._config[section]

            self._logger.assertIsInstance(
                sec_attributes,
                f"{section}_section",
                Dict[str, Any],
            )

            self._logger.indent()
            self._checkValidityAndRecastContent(section, sec_template["content"])
            self._logger.unindent()

        self._logger.debug(
            f"After check and recast, configuration reads\n{printDict(self._config, embedded=True)}"
        )

    # utils for the data hook

    def _readTxtData(self, filepath):
        """
        Load data in the txt format with np.genfromtxt()

        Columns should be separated by tabs or spaces. They should all have the same length.

        The header of the file can be arbitrary long, but the last row specifies the names of the
        columns.

        To store a vector named `myvector` of dim `n`, name the columns `myvector[0], ..., myvector[n-1]`.
        """
        file = open(filepath, "r")

        # counting lines in header
        len_header = 0
        while file.readline()[0] == "#":
            len_header += 1

        # reading the data
        data = np.genfromtxt(
            filepath,
            dtype=None,
            skip_header=len_header - 1,
            names=True,
            deletechars=" !#$%&'()*+, -./:;<=>?@\\^{|}~",
        )

        # transforming to dict
        data = {t: data[t] for t in data.dtype.fields}

        self._logger.Assert(
            "mask" not in data,
            ValueError(
                "`mask` is not a valid column name (it clashes with the internally defined mask)"
            ),
        )
        # looking for vectors in the keys
        veckeys = []
        for k in list(data.keys()):
            if "[" in k:  # ]
                w = k.find("[")  # ]
                vk = k[:w]
                if vk in veckeys:
                    continue
                # We found one columns, let's search for the others
                dim, vector = 0, []
                while True:
                    try:
                        vector.append(data[f"{vk}[{dim}]"])
                    except KeyError:
                        break
                    dim += 1
                data[vk] = np.stack(vector)
                veckeys.append(vk)

        for k in list(data.keys()):
            for vk in veckeys:
                if vk in k and vk != k:
                    del data[k]
                    break

        return data

    def _aliasData(
        self, name: str, aliases: Dict[str, str], data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Dict[str, str] = {old: new} of names of data to alias.
        """
        for old, new in aliases.items():
            self._logger.Assert(
                new not in data,
                ValueError(
                    f"In data item {name}, aliasing {old} to {new} clashes with existing data {new} "
                ),
            )
            self._logger.Assert(
                old in data,
                ValueError(
                    f"In data item {name}, cannot alias {old}, in not in {','.join(data.keys())}"
                ),
            )
            data[new] = data.pop(old)

        return data

    def _ignoreData(
        self, name: str, ignore: List[str], data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        List of strings.

        Ignore some columns. This is done after filtering. These columns will not be saved or
        accessible to the variables, modules, etc.

        Wildcards can be specified with '*pattern*', but no real regex is applied: 'a*b*c' is the
        same as '*abc*' or '*abc', etc...
        """

        wildcards = [ign.replace("*", "") for ign in ignore if "*" in ign]
        for k in data:
            if k in ignore or any((wc in k) for wc in wildcards):
                del data[k]

        return data

    def _filterData(
        self, name: str, filters: Dict[str, List], data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Quick filtering of the data. This is not meant to be a powerful filter, just a quick snippet
        to avoid having to multiply the number of data files when debugging.

        The filters have the form:

        catalog_name:
            path: path/to/file.txt
            filters:
                filter_1:
                    - key_1, val_1
                    - key_2, min_2, max_2
                    - key_4, [item_1, item_2, item_3]
                filter_2:
                    - key_1, val_1_bis,
                    - key_3, val_3
                    - key_4, min_4, max_4
                    - ...
                ...

        The mask constructed is:
            mask = (
                    (key_1 == val_1) AND
                    (min_2 < key_2 < max_2)) AND
                    (key_4 in [item_1, item_2, item_3]
                   )
                    OR
                   (
                    (key_1 == val_1_bis) AND
                    (key_3 == val_3) AND
                    (min_4 < key_4 < max_4) AND
                    (...)
                   )
                    OR
                   (
                    ...
                   )

        After filtering the data has len = sum(mask > 0), and the mask is added under the key 'mask'.

        """

        # the full mask
        or_mask = np.zeros(list(data.values())[0].shape[-1], bool)

        self._logger.assertIsInstance(filters, f"data:{name}:filters", Dict[str, List])

        # iterating over the filters
        for fname, fattrs in filters.items():
            self._logger.assertIsInstance(fattrs, f"data:{name}:filters:{fname}", List)

            # mask of the current filter
            and_mask = np.ones_like(or_mask)

            for i, fattr in enumerate(fattrs):
                self._logger.assertIsInstance(
                    fattr, f"data:{name}:filters:{fname}[{i}]", List
                )

                # only key == val or min < key < max
                self._logger.Assert(
                    len(fattr) in (2, 3),
                    ValueError(
                        f"Filter argument data:{name}:filters:{fname}[{i}] = {fattr} "
                        f"of length {len(fattr)} must be either a pair \n"
                        "   [key, value] for data[key] == value \n"
                        "   [key, min ,max] for min < data[key] < max"
                    ),
                )

                key = fattr[0]
                dat = self._getKeysFromDicts(key, **{f"data:{name}:loaded_data": data})

                # key in list case
                if len(fattr) == 2 and isinstance(fattr[1], (list, tuple)):
                    key_mask = np.isin(dat, fattr[1])  # type: ignore
                    self._logger.debug(
                        f"Selecting data:{name}[{key}] in list {fattr[1]}, "
                        f"{np.sum(key_mask)}/{key_mask.size} entries are kept"
                    )

                # key == val case
                elif len(fattr) == 2:
                    value = fattr[1]
                    key_mask = dat == value
                    and_mask *= key_mask

                    self._logger.debug(
                        f"Croping data:{name}[{key}] to value {value}, "
                        f"{np.sum(key_mask)}/{key_mask.size} entries are kept"
                    )

                # min < key < max case
                else:
                    _min, _max = float(fattr[1]), float(fattr[2])
                    key_mask = (dat > _min) * (dat < _max)  # type: ignore

                    self._logger.debug(
                        f"Croping data:{name}[{key}] to range {_min, _max}, "
                        f"{np.sum(key_mask)}/{key_mask.size} entries are kept"
                    )

                self._logger.debug(
                    f"data:{name}:filters:{fname} kept "
                    f"{np.sum(and_mask)}/{and_mask.size} entries"
                )
                and_mask *= key_mask

            # Adding "and_mask" to the "or_mask"
            or_mask += and_mask

        self._logger.debug(
            f"data:{name}:filters kept {np.sum(or_mask)}/{or_mask.size} entries"
        )

        # filtering the data
        for key, val in data.items():
            data[key] = val[..., or_mask]

        # adding the mask
        data[f"{name}_mask"] = or_mask

        return data

    def _loadDataItem(self, name, specs):
        """
        To be run on each data item.
        Loads and filters the data of the data item.
        """

        # if path is provided we load the data
        if "path" in specs:
            filepath: str = specs["path"]

            # npy format
            if filepath[-4:] == ".npy":
                _name = getKeysFromDicts(
                    "name", **{f"data:{name}:specs": specs}, logger=self._logger
                )
                dat = {_name: np.load(filepath, allow_pickle=False)}  # type: ignore

            # bin format
            elif filepath[-4:] == ".bin":
                format, shape = getKeysFromDicts(
                    "format",
                    "shape",
                    **{f"data:{name}:specs": specs},
                    logger=self._logger,
                    error_not_found=[True, False],
                )
                dat = {name: np.fromfile(filepath, dtype=format)}

                if shape is not None:
                    dat[name] = np.reshape(dat[name], shape)  # type: ignore

            # txt format
            elif filepath[-4:] == ".txt":
                dat = self._readTxtData(filepath)

            # we try with pandas
            else:
                ext = filepath[filepath.rfind(".") + 1 :]
                reader_name = f"read_{ext}"
                self._logger.Assert(
                    hasattr(pd, reader_name),
                    AttributeError(f"Cannot read {ext} file with pandas"),
                )
                dat = {
                    k: np.array(v)
                    for k, v in getattr(pd, reader_name)(filepath).items()
                }

        # Not used for now, but if you're good enough in yaml, you can provide the dict
        # only dict[str, np.ndarray] are accepted
        elif "raw" in specs:
            self._logger.assertIsInstance(specs["raw"], Dict[str, np.ndarray])  # type: ignore
            dat = specs["raw"]
        else:
            self._logger.error(
                ValueError(
                    f"Could not load anything from {name}, "
                    "please provide 'path' or 'data', "
                    f"got:\n{printDict(specs, embedded=True)}"
                )
            )

        self._logger.assertIsInstance(dat, f"{name}:loaded_data", Dict[str, np.ndarray])

        # Handling strings
        _dat = {}
        for k, v in dat.items():
            if "|S" in str(v.dtype):
                _dat[k] = v.astype(str)
        dat.update(_dat)

        # Running the filters
        if "filters" in specs:
            dat = self._filterData(name, specs["filters"], dat)  # type: ignore

        # Removing some columns
        if "ignore" in specs:
            dat = self._ignoreData(name, specs["ignore"], dat)  # type: ignore

        # Aliasing some columns
        if "aliases" in specs:
            dat = self._aliasData(name, specs["aliases"], dat)  # type: ignore

        # prepend
        if "prepend_name" in specs and specs["prepend_name"] is True:
            dat = {f"{name}_{k}": v for k, v in dat.items()}

        return dat

    def _loadDataFromConfig(self, data):
        """
        Loading, filtering each data item.
        See _loadDataItem and _filterData.
        """
        out_data = {}
        for data_key, data_item in data.items():
            item_out_data = self._loadDataItem(data_key, data_item)
            item_keys = list(item_out_data.keys())
            keys = list(out_data.keys())

            if len(keys):
                self._logger.assertIsNotIn(
                    item_keys,
                    [f"data:loaded_data:{data_key}[{k}]" for k in item_keys],
                    keys,
                    [f"already_loaded_data:[{k}]" for k in keys],
                )
            out_data.update(item_out_data)
        return out_data

    # Hooks to run on the config (re-formatting, loading data, etc)

    def _hookParameters(self, parameters):
        """
        Recursive check format on parameters.

        Accepted types are:
            Dict[str, Optional[Union[bool, int, float, str, List, Tuple, Dict]]],
        """

        #  recursive type checking of parameters
        def _check(dic, name):
            self._logger.assertIsInstance(
                dic,
                name,
                Dict[str, Optional[Union[bool, int, float, str, List, Tuple, Dict]]],
            )
            for k, v in dic.items():
                if isinstance(v, dict):
                    _check(v, f"{name}:{k}")

        _check(parameters, "parameters")
        return parameters

    def _hookData(self, data):
        """
        Loading, filtering each data item. Saving loaded data if `copy_data = True`.
        See _loadDataItem and _filterData.
        """
        ret_data = {}
        if self._config["general"]["load_working_dir"]:
            try:
                # I'll admit I am juggling between ConfigIO and IO
                # and it's not very very beautiful...
                ret_data = self._io.loadDataFromWorkingDir()
            except Exception as e:
                self._logger.warn(
                    "Could not load data from working dir, reverting to normal load. Got error\n"
                    + "\n".join(e.args),
                )

        self._logger.warn(f"{self._config['data']}")
        # We load the data in the config, might overwrite what's in the workding dir
        # we save also in the working dir
        if "data" in self._config:
            try:
                new_data = self._loadDataFromConfig(data)
            except TypeError as e:
                new_data = {}
                if ret_data == {}:
                    self._logger.error(
                        ValueError(
                            f"Could not load data from config. Got error\n"
                            + "\n".join(e.args)
                        )
                    )
                else:
                    self._logger.debug(
                        f"Could not load data from config. Got error\n"
                        + "\n".join(e.args)
                    )
            ret_data.update(new_data)
            # Save the data (maybe?)
            # if we managed to load from working dir, we're not saving again
            # Libraries need to be saved later, after they have been loaded
            if self._config["general"]["copy_data"]:
                self._logger.info("Saving data in the working dir")
                for name, dat in new_data.items():
                    self._io.saveNpy("data", name, dat, allow_pickle=False)

        return ret_data

    def runHooks(self):
        """
        Running the hooks (load data, check format of parameters)
        """
        for section in self._template_config.keys():
            if section in self._config:
                try:
                    hook = getattr(self, f"_hook{section.capitalize()}")
                except AttributeError:
                    continue
                self._config[section] = hook(self._config[section])

    # Interface core_attributes

    def toCoreAttributes(self) -> Dict[str, Any]:
        """
        Transforming configuration to attributes of the BananaCore object.
        Parsing self._config_core_map:
            True -> set as core attribute (after prepending with '_' to make it private)
            False -> ignored
            str -> name changed to str
            other = str -> unnamed attributes set in a dict of name str
        """
        core_attributes = {}
        for section, sec_items in self._config_core_map.items():
            # pop other if it's in sec_items
            try:
                sec_attributes = self._config[section]
            except KeyError:
                pass
            else:
                if sec_items is True:
                    core_attributes["_" + section] = sec_attributes
                elif sec_items is False:
                    continue
                elif isinstance(sec_items, str):
                    core_attributes[sec_items] = sec_attributes
                elif isinstance(sec_items, dict):
                    try:
                        other = sec_items.pop("other")
                        assert isinstance(other, (bool, str))
                    except KeyError:
                        other = None
                    for config_key, core_key in sec_items.items():
                        if core_key is False:
                            continue
                        if core_key is True:
                            # make it private
                            core_key = "_" + config_key
                        assert isinstance(config_key, str)
                        core_attributes[core_key] = sec_attributes[config_key]

                    # put the rest in "other" dict
                    if other is not None:
                        core_attributes[other] = {
                            k: v
                            for k, v in sec_attributes.items()
                            if k not in sec_items
                        }

        return core_attributes

    # def fromCoreAttributes(self, core_attributes: Dict[str, Any]) -> None:
    #     """
    #     Inverse function of self.toCoreAttributes.
    #     """

    #     def getKeyCore(key):
    #         try:
    #             return core_attributes[key]
    #         except KeyError:
    #             self._logger.warn(f"Core does not have attribute {key}")

    #     for section, sec_items in self._config_core_map.items():
    #         self._config[section] = {}
    #         if isinstance(sec_items, dict):
    #             try:
    #                 other = sec_items.pop("other")
    #                 assert isinstance(other, str)
    #             except KeyError:
    #                 other = None
    #             for config_key, core_key in sec_items.items():
    #                 if core_key is False:
    #                     continue
    #                 if core_key is True:
    #                     core_key = "_" + config_key
    #                 assert isinstance(config_key, str)
    #                 self._config[section][config_key] = getKeyCore(core_key)

    #             # unpack the rest from "other" dict
    #             if other is not None:
    #                 self._config[section].update(core_attributes[other])
    #         elif isinstance(sec_items, str):
    #             self._config[section] = getKeyCore(sec_items)
    #         elif sec_items == True:
    #             self._config[section] = getKeyCore("_" + section)


class IO:
    """
    All the IO of Banana, more or less.

    The ConfigIO is an attribute of this class, Core never directly communicates with it.
    """

    debug_dir = "./"

    def __init__(
        self,
        verbosity: int = 2,
    ):
        self._logger = Logger("IO", verbosity)
        self._config_io = ConfigIO(self, verbosity)

    def _setWorkingDir(self, working_dir: str):
        """
        Sets the working dir and calls self._constructWorkingDir.
        """
        self._working_dir = working_dir
        self._constructWorkingDir()

    def _constructWorkingDir(self):
        """
        Maybe populates the working dir with 'data', 'results', 'debug' and 'analysis' subdirs.

        Creates the attributes of IO, 'data_dir', 'res_dir', 'debug_dir' and 'anal_dir' which
        contain the full paths to the according directories.
        """
        # Any subdir would do
        populate = not isdir(join(self._working_dir, "analysis"))

        if populate:
            makedirs(self._working_dir)
            self._logger.debug("Populating dir")
        for attr, subdir in zip(
            ["data", "res", "debug", "anal"], ["data", "results", "debug", "analysis"]
        ):
            subpath = join(self._working_dir, subdir)
            setattr(self, attr + "_dir", subpath)
            if populate:
                makedirs(subpath)

    # load / save core

    def initCore(
        self,
        config: Union[str, Dict[str, Any]],
        argv: List[str],
    ) -> Dict[str, Any]:
        """
        Loads the core attributes from a configuration (dict or filepath).

        This function handles the ConfigIO, loads the configuration, check the types, casts and runs
        the hooks (i.e. the data is loaded and filtered here).

        It also creates the working dir and sets the LogServer output and highlights.

        What happens here:
            1. Load config with yaml if `config = path`, else set if dict
            2. If `load_working_dir = True`, load working_dir/config.yml and merge with given config
                - Crashes if it does not exist
            3. Set and populate working_dir, set verbosity
            4. Check the validity of the configuration and recast some arguments
            5. Set the highlighting of the logger
            6. Check the format of the parameters (no "weird" type)
            7. Load the data (and save it if `copy_data = True`)
            8. Save config.yml to working_dir
            9. Re-format to core attributes and return to core
        """

        # Recursive load
        self._config_io.recursiveLoad(config, argv)

        tmp_config = self._config_io.getConfig()
        # We try our luck to get the working dir, in case the rest of the check fails
        # If the "try" fails, the proper error message will be raised with "checkValidityAndRecast"
        try:
            working_dir = tmp_config["general"]["working_dir"]
            verbosity = tmp_config["logging"]["verbosity"]
        except KeyError:
            # That's bad news, but it'll crash properly later
            pass
        else:
            self._setWorkingDir(working_dir)
            self._logger.setWorkingDir(working_dir)

            self._logger.success(
                f"Working directory has been set to {self._working_dir} and "
                f"logger has started logging (with verbosity {verbosity})"
            )

        # In any case, we check the validity
        self._config_io.checkValidityAndRecast()

        config = self._config_io.getConfig()

        # we set the highlights of the logging
        # we needed to wait for the casting
        highlights = config["logging"]["highlights"]
        self._logger.setHighlights(highlights)

        # Let's run the hooks
        self._config_io.runHooks()

        # maybe saving the configuration (not overwritting if existing one)
        if not config["general"]["load_working_dir"]:
            self._config_io.dump(f"{working_dir}/config.yml")

        return self._config_io.toCoreAttributes()

    def saveCore(
        self,
        core_attributes: Dict[str, Any],
        config_file: Optional[str] = None,
        save_data: bool = True,
    ) -> None:
        """
        Saves the core attributes to the configuration file format.

        To avoid unnecessary io, set the `save_data` switch to False if the data has already been
        saved.
        """
        if save_data:
            for k, v in core_attributes["_data"].items():
                self.saveNpy("data", k, v)
        core_attributes["_data"] = list(core_attributes["_data"].keys())

        if config_file is None:
            config_file = join(self._working_dir, "config.yml")

        self._config_io.fromCoreAttributes(core_attributes)
        self._config_io.dump(config_file)

    def saveNpy(
        self,
        subdir: str,
        name: str,
        data: np.ndarray,
        allow_pickle: bool = False,
        save_mode: str = "overwrite",
    ) -> None:
        """
        Saves to npy file 'self.[subdir]_dir/name.npy'
        """

        file = f"{getattr(self, f'{subdir}_dir')}/{name}.npy"
        self._logger.info(f"Saving data {name} in {file}")
        if os.path.isfile(file):
            if save_mode == "overwrite":
                self._logger.warn(f"{file} exists, overwriting it")
                pass
            elif save_mode == "safe":
                raise FileExistsError(f"{file} exists, will not overwrite!")
            elif save_mode == "append":
                self._logger.info(f"{file} exists, appending new data to stored")
                data = np.concatenate(
                    [np.load(file, allow_pickle=allow_pickle), data], axis=0
                )
        np.save(file, data, allow_pickle=allow_pickle)

    def loadNpy(
        self,
        subdir: str,
        name: str,
        allow_pickle: bool = False,
        error_not_found: bool = True,
        warn_not_found: bool = True,
        return_not_found: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Loads from npy file 'self.[subdir]_dir/name.npy'
        """
        file = f"{getattr(self, f'{subdir}_dir')}/{name}.npy"
        self._logger.info(f"Loading data {name} from {file}")
        try:
            return np.load(file, allow_pickle=allow_pickle)
        except FileNotFoundError:
            if error_not_found:
                raise
            elif warn_not_found:
                self._logger.warn(f"{file} not found")
            return return_not_found

    def saveDill(self, subdir: str, name: str, data: Any) -> None:
        """
        Saves to dill file 'self.[subdir]_dir/name.dill'
        """

        file = f"{getattr(self, f'{subdir}_dir')}/{name}.dill"
        self._logger.info(f"Saving data {name} to {file}")
        with open(file, "wb") as f:
            dill.dump(data, f)

    def loadDill(self, subdir: str, name: str) -> Any:
        """
        Loads from dill file 'self.[subdir]_dir/name.dill'
        """

        file = f"{getattr(self, f'{subdir}_dir')}/{name}.dill"
        self._logger.info(f"Loading data {name} from {file}")
        with open(file, "rb") as f:
            data = dill.load(f)
        return data

    #
    def loadDataFromWorkingDir(self):
        # already set at this time

        # Should never append...
        self._logger.Assert(
            isdir(self.data_dir), FileNotFoundError(f"{self.data_dir} does not exist")
        )

        # listing the files
        files = [f for f in os.listdir(self.data_dir) if isfile(join(self.data_dir, f))]

        # building the data dict
        data = {}
        for file in files:
            self._logger.Assert(
                file[-4:] == ".npy",
                FileNotFoundError(f"{file} is not .npy, cannot be loaded"),
            )
            name = file.replace(".npy", "")
            data[name] = self.loadNpy("data", name)

        return data
