from typing import Union, Any, Optional, List, Tuple, Dict
from types import ModuleType
from functools import partial
from tqdm import tqdm, trange

import jax
import jax.random as jra
import jax.numpy as jnp
from jax._src.tree_util import tree_map
from jax._src.public_test_util import (
    numerical_jvp,
    conj,
    inner_prod,
    rand_like,
)
from jax.scipy.ndimage import map_coordinates
import numpy as np

from tensorflow_probability.substrates import jax as tfp
import inspect

from .common import fdtype, idtype, cdtype, Jnp_ndarray, TypingType


##############
### Basics ###
##############


def toList(obj: Union[Any, List[Any], Tuple[Any]]) -> List[Any]:
    return list(obj) if isinstance(obj, (list, tuple)) else [obj]


def atLeast2DLastAxis(
    arr: Union[np.ndarray, Jnp_ndarray]
) -> Union[np.ndarray, Jnp_ndarray]:
    if arr.ndim == 1:
        return arr.reshape((arr.size, 1))
    else:
        return arr


##################
### TF helpers ###
##################


def f_cast(data: Any) -> Jnp_ndarray:
    """

    Parameters
    ----------
    data: Any :


    Returns
    -------


    """
    return jnp.asarray(data, dtype=fdtype)


def c_cast(data: Any) -> Jnp_ndarray:
    """

    Parameters
    ----------
    data: Any :


    Returns
    -------


    """
    return jnp.asarray(data, dtype=cdtype)


def i_cast(data: Any) -> Jnp_ndarray:
    """

    Parameters
    ----------
    data: Any :


    Returns
    -------


    """
    return jnp.asarray(data, dtype=idtype)


def safeDivide(num, denom):
    # avoiding nans
    mask = denom > 0
    return f_cast(mask) * num / jnp.where(mask, denom, 1.0)


###########################################
### TF Multiplication between complexes ###
###########################################


def cmulriri(ar, ai, br, bi):
    """

    Parameters
    ----------
    ar :

    ai :

    br :

    bi :


    Returns
    -------


    """
    cr = ar * br - ai * bi
    ci = ar * bi + ai * br
    return (cr, ci)


def cmulrir(ar, ai, br):
    """

    Parameters
    ----------
    ar :

    ai :

    br :


    Returns
    -------


    """
    cr = ar * br
    ci = ai * br
    return (cr, ci)


def cmulrii(ar, ai, bi):
    """

    Parameters
    ----------
    ar :

    ai :

    bi :


    Returns
    -------


    """
    cr = -ai * bi
    ci = ar * bi
    return (cr, ci)


################################
### Compute Hessian with jit ###
################################


def hessian(f, x):
    """

    Parameters
    ----------
    f :

    x :


    Returns
    -------


    """
    raise NotImplementedError("TODO")


###########################
### Time and statistics ###
###########################


def time2hms(t):
    """

    Parameters
    ----------
    t :


    Returns
    -------


    """
    h = int(t / 3600)
    m = int((t - h * 3600) / 60)
    s = int((t - h * 3600 - m * 60))
    return f"{h}h:{m}m:{s}s"


@partial(jax.jit, static_argnames=["collapsed"])
def Rsplit(data, collapsed=True):
    data = atLeast2DLastAxis(data)
    n = data.shape[0]

    mean0 = jnp.mean(data[: n // 2, :], axis=0)
    mean1 = jnp.mean(data[n // 2 :, :], axis=0)
    mean_sq0 = jnp.mean(data[: n // 2, :] ** 2, axis=0)
    mean_sq1 = jnp.mean(data[n // 2 :, :] ** 2, axis=0)
    std0 = jnp.sqrt(jnp.maximum(mean_sq0 - mean0**2, 0))
    std1 = jnp.sqrt(jnp.maximum(mean_sq1 - mean1**2, 0))

    mean = 0.5 * (mean0 + mean1)
    mean_sq = 0.5 * (mean_sq0 + mean_sq1)
    std = jnp.sqrt(jnp.maximum(mean_sq - mean**2, 0))

    if collapsed:
        return jnp.mean(0.5 * (std0 + std1) / std), jnp.mean(mean), jnp.mean(std)
    else:
        return 0.5 * (std0 + std1) / std, mean, std


@partial(jax.jit, static_argnames=["ess"])
def chainSummaryStatistics(chain, prior_mean, prior_cov, ess=True):
    # post_mean = jnp.mean(chain, axis=0)
    # post_cov = jnp.sqrt(jnp.mean(chain**2, axis=0) - post_mean**2)
    r_split, post_mean, post_cov = Rsplit(chain, collapsed=False)

    if ess:
        chain_ess = tfp.mcmc.effective_sample_size(
            chain, filter_beyond_positive_pairs=True
        )
        chain_ess = jnp.minimum(chain_ess, chain.shape[0])
    else:
        chain_ess = -1 * jnp.ones_like(prior_mean)

    ratio_mean = safeDivide(post_mean, prior_mean)
    ratio_cov = safeDivide(post_cov, prior_cov)
    signi = safeDivide(post_mean - prior_mean, post_cov)

    return post_mean, post_cov, ratio_mean, ratio_cov, signi, r_split, chain_ess


def descriptiveStatistics(
    data: np.ndarray,
    collapse_above: int = 10,
    reference: Optional[Union[List[float], List[List[float]]]] = None,
    coloring: bool = False,
    mod: Any = np,
    return_err: bool = False,
):
    """

    Parameters
    ----------
    data: np.ndarray :

    collapse_above: int :
         (Default value = 10)
    reference: Optional[Union[List[float] :

    List[List[float]]]] :
         (Default value = None)
    coloring: bool :
         (Default value = False)
    mod: Any :
         (Default value = np)
    return_err: bool :
         (Default value = False)

    Returns
    -------


    """

    data = np.atleast_1d(data)
    # Transforming 1-shape array to float (or int...)
    if data.ndim == 1 and data.size == 1:
        data = data[0]

    if data.ndim > 0:
        mean = mod.mean(data, axis=0)
        std = mod.std(data, axis=0)
        min = mod.min(data, axis=0)
        max = mod.max(data, axis=0)
        non_finite = mod.sum(~mod.isfinite(data))
        if data.shape[-1] > collapse_above:
            mean = mod.mean(mean)
            std = mod.mean(std)
            min = mod.min(min)
            max = mod.max(max)
    else:
        mean = min = max = data
        std = 0.0
        non_finite = ~mod.isfinite(data)

    mean, std, min, max = [float(np.squeeze(x)) for x in [mean, std, min, max]]
    non_finite = int(non_finite)

    ret = ""
    if reference is not None and isinstance(mean, np.ndarray):
        raise NotImplementedError(
            "Not collapsed or np.ndarray data and reference are not compatible"
        )
    elif isinstance(mean, np.ndarray):
        ret += f"{'min ':.<20s} {min}\n"
        ret += f"{'max ':.<20s} {max}\n"
        ret += f"{'mean ':.<20s} {mean}\n"
        ret += f"{'std ':.<20s} {std}\n"
    elif reference is None:
        ret += f"{'min ':.<20s} {min:10f}\n"
        ret += f"{'max ':.<20s} {max:10f}\n"
        ret += f"{'mean ':.<20s} {mean:10f}\n"
        ret += f"{'std ':.<20s} {std:10f}\n"
    else:
        assert len(reference) == 5, "refence = [min, max, mean, std]"
        for key, val, ref in enumerate(
            zip(
                ["min", "max", "mean", "std"],
                [min, max, mean, std],
                reference,
            )
        ):
            if len(ref) == 1:
                ret += f"{key:.<20s} {val:10f} (expected {ref[0]:10f})\n"
            elif len(ref) == 2:
                ret += f"{key:.<20s} {val:10f} (expected value within {ref[0]:10f} and {ref[1]:10f})\n"
            elif len(ref) == 4:
                ret += f"{key:.<20s} {val:10f} (expected value within {ref[1]:10f} and {ref[2]:10f}, is very wrong outside {ref[0]:10f} and {ref[3]:10f})\n"
            else:
                raise ValueError(
                    "Reference is either [expected] or [low, high] or [error_low, warning_low, warning_high, error_high]"
                )
    ret += f"{'non finite ':.<20s} {non_finite:10d}"
    if return_err:
        return ret, non_finite > 0
    else:
        return ret


def printTable(
    rows: List[str],
    columns: List[str],
    data: np.ndarray,
    data_fmt: str = "%15.5f",
    embedded: bool = True,
):
    """Builds a beautiful formated table to be printed out"""
    assert isInstance(columns, "columns", List[str])[0], f"Got columns = {columns}"
    assert isInstance(rows, "rows", List[str])[0], f"Got rows = {rows}"
    assert isinstance(data, np.ndarray), f"Got data = {type(data)}"
    assert data.shape == (
        len(rows),
        len(columns),
    ), f"Got {data.shape=} vs (ncols={len(columns)}, nrows={len(rows)})"

    len_cell = len(data_fmt % (0.0))  # dummy input

    len_col = 2 + max([max([len(col) for col in columns]), len_cell])
    len_row = 2 + max([len(col) for col in columns])

    col_fmt = f"%{len_col}s"
    row_fmt = f"%{len_row}s"

    _data_fmt = data_fmt.replace(str(len_cell), str(len_col))
    assert _data_fmt != data_fmt, "Only fixed sized input are valid (%15.5f) not (%.5f)"

    to_print = row_fmt % ("") + "".join([col_fmt % (col) for col in columns])
    for row, dat in zip(rows, data):
        line = row_fmt % (row) + "".join([_data_fmt % (d) for d in dat])
        to_print += "\n" + line

    if not embedded:
        print(to_print)
    else:
        return to_print


def descriptiveStatisticsTable(
    functions: Tuple[str] = ("min", "max", "mean", "std"),
    names: Optional[Dict[str, str]] = None,
    math: ModuleType = np,
    max_len_unwrap: int = 5,
    **data,
):
    def one_line(name, dat):
        if dat.size == 1:
            try:
                _str = f"{float(np.squeeze(dat)):15.5f}"
            except ValueError:
                _str = f"{str(dat):15s}"
            return f"{name:<30s}" + " " * 15 * 2 + _str + " " * 15 + f"{1:>20d}\n"
        else:
            values = []
            for f in math_func:
                try:
                    val = f(dat)
                except (np.core._exceptions.UFuncTypeError, ValueError):  # type: ignore
                    val = np.nan

                values.append(val)
            return (
                f"{name:<30s}"
                + "".join(f"{v:15.5f}" for v in values)
                + f"{','.join(str(s) for s in dat.shape):>20s}\n"
            )

    math_func = [getattr(math, f) for f in functions]
    to_print = (
        f"{'':<30s}" + "".join(f"{f:>15s}" for f in functions) + f"{'shape':>20s}\n"
    )
    names = {} if names is None else names
    assert isinstance(names, dict)
    for k, dat in data.items():
        name = names.get(k, k)
        try:
            dat = jnp.atleast_1d(dat)
        except TypeError:
            to_print += f"{name}: no valid statistics on this data type\n"
            continue
        if len(dat) > 1 and len(dat) < max_len_unwrap:
            for i, _dat in enumerate(dat):
                _name = f"{name}[{i}]"
                to_print += one_line(_name, _dat)
        else:
            to_print += one_line(name, dat)

    return to_print


def describe(y, x=None, dx=10, logger=None, xlabel=None, ylabel=None):
    """

    Parameters
    ----------
    y :

    x :
         (Default value = None)
    dx :
         (Default value = 10)
    logger :
         (Default value = None)
    xlabel :
         (Default value = None)
    ylabel :
         (Default value = None)

    Returns
    -------


    """
    string = ""
    if x is not None:
        bins = np.arange(0, np.max(x), dx)
        string += "%15s %15s %15s %15s %15s %15s %15s %15s\n" % (
            "from",
            "to",
            "min",
            "mean",
            "median",
            "max",
            "std",
            "count",
        )
        out = np.zeros((7, len(bins) - 1))
        for i, (l, r) in enumerate(zip(bins[:-1], bins[1:])):
            w = (x > l) * (x < r)
            _y = y[w]
            if len(_y) == 0:
                continue
            out[0, i] = (l + r) / 2
            out[1:, i] = (
                np.min(_y),
                np.mean(_y),
                np.median(_y),
                np.max(_y),
                np.std(_y),
                _y.size,
            )
            string += "%15.5g %15.5g %15.5g %15.5g %15.5g %15.5g %15.5g %15i\n" % (
                (l, r) + tuple(out[1:, i])
            )
    else:
        string += "%15s %15s %15s %15s %15s %15s\n" % (
            "min",
            "mean",
            "median",
            "max",
            "std",
            "count",
        )
        out = np.array(
            [np.min(y), np.mean(y), np.median(y), np.max(y), np.std(y), y.size]
        )
        string += "%15.5g %15.5g %15.5g %15.5g %15.5g %15i\n" % tuple(out)
    if logger is not None and x is None:
        logger.info(f"{ylabel} shows statistics:\n{string}")
    elif logger is not None:
        logger.info(f"{ylabel} vs {xlabel} shows statistics:\n{string}")
    return out


#############################
### Dictionaries handling ###
#############################


def getKeysFromDicts(
    *keys: Union[str, List[str]],
    logger=None,  # type Logger not imported to avoid import loop
    error_several_occurences: Union[bool, List[bool]] = True,
    error_not_found: Union[bool, List[bool]] = True,
    not_found_return_value: Union[Any, List[Any]] = None,
    return_dict: bool = False,
    recursive: bool = False,
    **dicts: dict,
):  # dropped typing here as it can return too many different things
    """

    Parameters
    ----------


    Returns
    -------


    """
    assert logger is not None

    if recursive:
        next_call_recursive = False
        inner_dicts = {}
        for dictname, _dict in dicts.items():
            for k, v in _dict.items():
                if isinstance(v, dict):
                    new_k = f"{dictname}_{k}"
                    if new_k not in dicts:
                        inner_dicts[new_k] = v
                        next_call_recursive = True

        return getKeysFromDicts(
            *keys,
            logger=logger,
            error_several_occurences=error_several_occurences,
            error_not_found=error_not_found,
            not_found_return_value=not_found_return_value,
            return_dict=return_dict,
            recursive=next_call_recursive,
            **dicts,
            **inner_dicts,
        )

    keys = [keys] if isinstance(keys, str) else keys
    error_several_occurences = (
        [error_several_occurences] * len(keys)
        if isinstance(error_several_occurences, bool)
        else error_several_occurences
    )
    error_not_found = (
        [error_not_found] * len(keys)
        if isinstance(error_not_found, bool)
        else error_not_found
    )
    not_found_return_value = (
        [not_found_return_value] * len(keys)
        if not (
            isinstance(not_found_return_value, list)
            and len(not_found_return_value) == len(keys)
        )
        else not_found_return_value
    )
    logger.assertShapesMatch(
        keys=[len(keys)],
        error_not_found=[len(error_not_found)],
        error_several_occurences=[len(error_several_occurences)],
    )
    vals = []
    for key, err_sev_occ, err_not_found, default in zip(
        keys, error_several_occurences, error_not_found, not_found_return_value
    ):
        val = default
        source = None
        for dict_name, _dict in dicts.items():
            try:
                _val, _source = _dict[key], dict_name
            except KeyError:
                pass
            else:
                if err_sev_occ and val is not default:
                    logger.error(
                        KeyError(
                            f"{key} has been found in (at least) two dictionaries: {source}:{val}, {_source}:{_val}"
                        )
                    )
                val, source = _val, _source
                break
        else:
            if err_not_found:
                str_dict = "\n".join(
                    f"{kdict}:{list(vdict.keys())}" for kdict, vdict in dicts.items()
                )
                logger.error(
                    KeyError(
                        f"{key} has not been found in any of the following dictionaries: \n{str_dict}"
                    )
                )
        vals.append(val)

    if return_dict:
        return {k: v for k, v in zip(keys, vals)}
    elif len(vals) == 1:
        vals = vals[0]
    return vals


def getKeysFromDictsWildCards(
    *keys: Union[str, List[str]],
    logger,  # type Logger not imported to avoid import loop
    error_not_found: Union[bool, List[bool]] = True,
    not_found_return_value: Any = None,
    **dicts: dict,
) -> Dict[str, Any]:
    """

    Parameters
    ----------
    *keys: Union[str :

    List[str]] :

    logger :

    # type Logger not imported to avoid import looperror_not_found: Union[bool :

    List[bool]] :
         (Default value = True)
    not_found_return_value: Any :
         (Default value = None)
    **dicts: dict :


    Returns
    -------


    """
    all_avail_keys = {name: list(_dict.keys()) for name, _dict in dicts.items()}

    ret = {}
    for key in keys:
        radkey = key.replace("*", "")
        for dict_name, dict_keys in all_avail_keys.items():
            vals = {k: dicts[dict_name][k] for k in dict_keys if radkey in k}
            if len(vals) > 0:
                ret.update(vals)
                break
        else:
            if error_not_found:
                str_dict = "\n".join(
                    f"{kdict}:{list(vdict.keys())}" for kdict, vdict in dicts.items()
                )
                logger.error(
                    KeyError(
                        f"{key} has not been found in any of the following dictionaries: \n{str_dict}"
                    )
                )
            else:
                vals = {key: not_found_return_value}
        ret.update(vals)
    return ret


def findInDicts(key, *dicts):
    """Searches for a key in several dictionaries and return its value. If the key appears in more than
    one dictionary, a KeyError is raised

    Parameters
    ----------
    key :

    *dicts :


    Returns
    -------

    """
    ret = None
    for _dict in dicts:
        if key in _dict:
            if ret is None:
                ret = _dict[key]
            else:
                raise KeyError(
                    f"{key} appears in several dictionaries: \n"
                    + "\n\n".join([str(_dict) for _dict in dicts])
                )
    return ret


def updateDictRecursive(
    dictmod, dictsrc, logger=None, inplace=True, limit_to_existing_keys=True
):
    """ """

    def log(message):
        if logger is not None:
            logger.debug(message)

    mkeys_with_wildcards = [k.replace("*", "") for k in dictmod if "*" in k]

    def keepKey(key):

        if key in dictmod:
            return True
        else:
            for mkey in mkeys_with_wildcards:
                if mkey in key:
                    return True
        return False

    # logger.debug(f"Dict to modify \n{printDict(dictmod, embedded=True)}")
    # logger.debug(f"Modifier dict \n{printDict(dictsrc, embedded=True)}")
    if not inplace:
        dictmod = dictmod.copy()
    if limit_to_existing_keys:
        dictsrc = {key: val for key, val in dictsrc.items() if keepKey(key)}
    for key, val in dictsrc.items():
        if isinstance(val, dict):
            log(f"Updating kwargs {key} as a dict")
            if key in dictmod:
                updateDictRecursive(
                    dictmod[key],
                    val,
                    logger=logger,
                    inplace=True,
                    limit_to_existing_keys=limit_to_existing_keys,
                )
            elif not limit_to_existing_keys:
                dictmod[key] = val
        else:
            log(f"Replacing value of {key} with {val}")
            dictmod[key] = val
    # logger.debug(f"Dict modified \n{printDict(dictmod, embedded=True)}")
    return dictmod


def printDict(_dict, embedded=False, indent=4, layer=0):
    """

    Parameters
    ----------
    _dict :

    embedded :
         (Default value = False)
    indent :
         (Default value = 4)
    layer :
         (Default value = 0)

    Returns
    -------


    """
    base_indent = " " * indent * layer
    item_indent = " " * indent * (layer + 1)
    subitem_indent = " " * indent * (layer + 2)
    if layer == 0:
        string = f"{base_indent}" + "{\n"
    else:
        string = ""
    for k, v in _dict.items():
        if isinstance(v, dict):
            string += f"{item_indent}'{k}':" + " {\n"
            string += printDict(v, embedded=True, indent=indent, layer=layer + 1)
        elif isinstance(v, (tuple, list)):
            string += f"{item_indent}'{k}': [" + "\n"
            for item in v:
                if isinstance(item, str):
                    _item = f"'{item}'"
                else:
                    _item = f"{item}"
                string += f"{subitem_indent}{_item}" + "\n"
            string += f"{item_indent}]" + "\n"
        else:
            if isinstance(v, str):
                _item = f"'{v}'"
            else:
                _item = f"{v}"
            string += f"{item_indent}'{k}': {_item}" + "\n"
    string += f"{base_indent}" + "}\n"
    if not embedded:
        print(string)
    else:
        return string


def replaceKeysInDict(dic, key_map, logger, error_not_found=False):
    """

    Parameters
    ----------
    dic :

    key_map :

    logger :

    error_not_found :
         (Default value = False)

    Returns
    -------


    """
    old_keys = list(key_map)
    if error_not_found:
        logger.assertIsIn(old_keys, old_keys, dic, "replaceKeysInDict_input_dic")
    for old, new in key_map.items():
        try:
            dic[new] = dic[old].pop()
        except KeyError:
            pass
    return dic


##########################
### Casting and typing ###
##########################


def printType(
    _type, embedded: bool = False, hide_modules: Union[str, List[str]] = "all"
) -> Union[str, None]:
    str_type = str(_type)

    # compliance with inspect.signature.parameters.annotation
    if "_empty" in str_type:
        return "no_type"

    # remove base types and typing. module extension, leaving the rest
    for sub_str in ["<class '", "'>"]:
        str_type = str_type.replace(sub_str, "")

    if hide_modules == "none":
        pass
    elif isinstance(hide_modules, list):
        for mod in hide_modules:
            str_type = str_type.replace(f"{mod}.", "")
    elif hide_modules == "all":
        # 'Optional[a/b/c.py.TYPE]' -> 'Optinal[TYPE]'
        # find a point and walk backward
        c = 0
        while (dot := str_type.rfind(".")) >= 0:
            pos = dot - 1
            while pos >= 0 and (str_type[pos].isalpha() or str_type[pos] in "./-_"):
                if c > 1000:
                    raise
                c += 1
                pos -= 1

            str_type = str_type[: pos + 1] + str_type[dot + 1 :]
        # str_type = str_type[str_type.rfind(".") + 1 :]
    else:
        raise ValueError(f"hide_modules not understood: {hide_modules}")

    if embedded:
        return str_type
    else:
        print(str_type)


def isInstance(
    obj: Any, name: str, _type: Union[type, Tuple[type], TypingType]
) -> Tuple[bool, str, str]:
    """
    Checks recursively the type of an object given a type from the 'typing' library
    """

    # print(f"{name=}")
    # print(f"{type(obj)=}")
    # print(f"{_type=}")

    # Any is a type but cannot be used in isinstance
    if _type == Any:
        return True, name, printType(_type, embedded=True)
    # simple case
    if isinstance(_type, (tuple, type)):
        return isinstance(obj, _type), name, printType(_type, embedded=True)

    # now getting into the typing
    # we give up on correct typing and go down the string way
    str_inst = str(_type)

    # Every type from typing is preceded by 'typing.'
    # we use that as a check
    if not ("typing" in str_inst):
        raise TypeError(
            f"The type {_type} to check against is either a base type nor from typing"
        )

    # Getting the first part of the type (until [])
    w = str_inst.find("[")  # ]
    w = w if w > 0 else len(str_inst)
    root = str_inst[:w].replace("typing.", "")

    # Checking the root
    if root == "Tuple":
        if not (ret := isInstance(obj, name, tuple))[0]:
            return ret
    elif root == "List":
        if not (ret := isInstance(obj, name, list))[0]:
            return ret
    elif root == "Dict":
        if not (ret := isInstance(obj, name, dict))[0]:
            return ret
    elif root == "Any":
        pass
    elif root in ("Union", "Optional"):
        pass
    else:
        raise NotImplementedError(
            "Only Tuple, List, Dict, Any, Optional and Union are understood. "
            f"Got instead {root}"
        )

    # Analysing the subtypes
    # Some implemetations / versions of typing seem to have '~T' / '~KT' as args anyway
    if hasattr(_type, "__args__") and not any(
        "~" in str(arg) for arg in _type.__args__
    ):
        # Lists and Tuples are done in the same way
        if root in ("List", "Tuple"):
            items_type = _type.__args__[0]
            for i, item in enumerate(obj):
                if not (ret := isInstance(item, f"{name}[{i}]", items_type))[0]:
                    return ret
        # Anylising every key and every value for a dict
        elif root == "Dict":
            keys_type = _type.__args__[0]
            values_type = _type.__args__[1]
            for k, v in obj.items():
                if not (ret := isInstance(k, f"{name}[{k}].key", keys_type))[0]:
                    return ret
                if not (ret := isInstance(v, f"{name}[{k}].value", values_type))[0]:
                    return ret
        # Optional: is None or check
        elif root == "Optional":
            subtype = _type.__args__[0]
            if obj is not None:
                if not (ret := isInstance(obj, name, subtype))[0]:
                    return ret
        # Union: running recursively on every value and catching errors
        elif root == "Union":
            for subtype in _type.__args__:
                if (ret := isInstance(obj, name, subtype))[0]:
                    return ret
            else:
                return False, name, printType(_type, embedded=True)
    elif root in ("Union", "Optional"):
        raise TypeError(
            "Union and Optional need to be provided with subtypes",
            " e.g. Union[str, int]",
        )

    return True, name, printType(_type, embedded=True)


def conditionalCastTo(obj, name, type_func_map, logger, error_uncasted=False):
    """
    type_func_map = tuple of pairs (type, func / value)

    For instance: (((int, float), lambda x: [x]), (None, []))
    """

    def _match(obj, _type):
        if isinstance(_type, tuple):
            return any(_match(obj, t) for t in _type)
        elif isinstance(_type, type):
            return isinstance(obj, _type)
        elif _type in (None, False, True):
            return obj is _type
        else:
            return obj == _type

    # (type, func) -> ((type, func))
    if not isinstance(type_func_map[-1], tuple):
        type_func_map = (type_func_map,)

    # iterate of over ((type, func))
    for _type, func_or_ret in type_func_map:
        # _type is a type or tuple thereof
        if _match(obj, _type):
            # is rhs a callable? Call it!
            if callable(func_or_ret):
                try:
                    return func_or_ret(obj)
                except Exception as e:
                    logger.error(
                        TypeError(
                            f"Could not cast {name} of type {type(obj)} identified to type "
                            f"{_type} with function {inspect.getsource(func_or_ret)}"
                            "Caught error reads:",
                            *e.args,
                        )
                    )
            # no? return rhs as such
            else:
                return func_or_ret

    else:
        if error_uncasted:
            logger.error(
                TypeError(
                    f"{name} of type {type(obj)} to was not found "
                    f"in the types map ({', '.join([t for t, _ in type_func_map])})"
                )
            )
        else:
            return obj


#########################
### USER INTERACTIONS ###
#########################


def stubbornInput(question: str, accepted_answers: List[str]) -> str:
    """

    Parameters
    ----------
    question: str :

    accepted_answers: List[str]


    Returns
    -------


    """
    while True:
        answer = input(question)
        if answer in accepted_answers:
            return answer
        else:
            print("Invalid answer...")


def getJaxRandomKey(seed=None):
    """

    Parameters
    ----------
    seed :
         (Default value = None)

    Returns
    -------


    """
    if seed is None:
        seed = np.random.randint(2**32)
    return jra.PRNGKey(seed)


#######################
### CHECK GRADIENTS ###
#######################


def _checkVJP(f, args, eps, use_tqdm, n_rand):
    f_vjp = partial(jax.vjp, f)
    _rand_like = partial(rand_like, np.random.RandomState(np.random.randint(2**32)))
    v_out, vjpfun = f_vjp(*args)

    all_ans = []
    for _ in (
        trange(n_rand, desc="Randomizing", leave=False) if use_tqdm else range(n_rand)
    ):
        tangent = tree_map(_rand_like, args)
        cotangent = tree_map(_rand_like, v_out)
        cotangent_out = list(conj(vjpfun(conj(cotangent))))
        ip = inner_prod(tangent, cotangent_out)

        ans = []
        iter = [eps] if isinstance(eps, float) else eps
        for _eps in (
            tqdm(iter, desc="Checking gradients", leave=False) if use_tqdm else iter
        ):
            tangent_out = numerical_jvp(f, args, tangent, eps=_eps)
            ip_expected = inner_prod(tangent_out, cotangent)
            ans.append(jnp.abs(ip / ip_expected - 1))

        all_ans.append(ans)

    ans = np.median(all_ans, axis=0)
    return v_out, ans[0] if isinstance(eps, float) else ans


def checkVJP(f, args, eps, use_tqdm=False, n_rand=1, n_out=None):
    n_in = len(args)

    if not n_out:
        v_out = f(*args)
        n_out = len(v_out) if isinstance(v_out, tuple) else 1

    inout_err = [[]]
    for i in trange(n_in, desc="input"):
        _err = []
        for j in trange(n_out, desc="output"):

            def partial_f(subargs):
                _args = list(args)
                _args[i] = subargs
                return f(*args)[j]

            v_out, err = _checkVJP(partial_f, args[i], eps, use_tqdm, n_rand)
            _err.append(err)
        inout_err.append(_err)

    return v_out, np.squeeze(inout_err)


### GRID UTILS ###


def subsample3DGrid(field: Jnp_ndarray, L_in: int, n_out: int, L_out: int):
    n_in = field.shape[0]
    if n_in == n_out and L_in == L_out:
        return field
    elif (L_in / L_out % 2 == 0) and (n_in / n_out % 2 == 0):
        ratio = L_in // L_out
        beg = n_in // 2 - ratio * n_in // 2
        end = n_in // 2 + ratio * n_in // 2
        step = (end - beg) // n_out
        if step == 0:
            raise ValueError(
                f"Cannot subsamble grid {n_in=} {L_in=} to {n_out=} {L_out=}"
            )

        sl = slice(beg, end, step)
        return field[(sl,) * 3]
    else:
        raise ValueError(f"Cannot subsamble grid {n_in=} {L_in=} to {n_out=} {L_out=}")


def interpolateOn3DGrid(
    field: Jnp_ndarray, positions: Jnp_ndarray, L_grid: fdtype
) -> Jnp_ndarray:
    """

    Parameters
    ----------
    field: Jnp_ndarray :

    positions: Jnp_ndarray :

    L_grid: fdtype :


    Returns
    -------


    """
    return map_coordinates(
        field, field.shape[0] / L_grid * (positions + L_grid / 2), order=1
    )
