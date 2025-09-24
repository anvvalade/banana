import time
from sys import stdout
from typing import Any, TypeVar, Union, Optional, List, Tuple, Callable, Dict
from types import ModuleType
from .common import jnp_ndarray, Jnp_ndarray, TypingType
from .utils import (
    printTable,
    descriptiveStatistics,
    descriptiveStatisticsTable,
    getKeysFromDicts,
    isInstance,
    printType,
    checkVJP,
    toList,
)

from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax


class ANSI:
    """ """

    # closing the [] for my linter
    OKBLUE = "\033[94m"  # ]
    INFO = "\033[96m"  # ]
    SUCCESS = "\033[92m"  # ]
    DEBUG = "\033[95m"  # ]
    WARNING = "\033[93m"  # ]
    ERROR = "\033[1m\033[91m"  # ]]
    HIGHLIGHT = "\033[1m\033[91m"  # ]]
    ENDC = "\033[0m"  # ]
    BOLD = "\033[1m"  # ]
    UNDERLINE = "\033[4m"  # ]


class UnknownError(Exception):
    """ """

    def __init__(self, *args):
        super().__init__(*args)


class ShapesMatchError(Exception):
    """ """

    def __init__(self, *args):
        super().__init__(
            f"Shapes of {args[0]} ({args[1]}) and {args[2]} ({args[3]}) should match"
        )


class LogServer:
    """Class LogServer:
        Server to which all logger instances connect.
        Change flush_every to flush more often in log file (it might be slower). In debug mode,
        every line is logged.

    Parameters
    ----------

    Returns
    -------

    """

    cannot_save_error = False
    flush_every = 10
    highlights = []
    _working_dir = None
    _indent_str = "    "
    _indent_level = 0

    def __init__(self, working_dir: Optional[str] = None) -> None:
        if working_dir is not None:
            self.setWorkingDir(working_dir)
        self._logged = []

    def __del__(self):
        self.printAndLog("LogServer", "SUCCESS", "Done logging, exciting properly!")
        self.flush()

    def setHighlights(self, highlights: List[str]) -> None:
        assert isinstance(highlights, list), highlights
        assert all(isinstance(h, str) for h in highlights), highlights
        self.highlights = highlights

    def setWorkingDir(self, working_dir: str) -> None:
        """

        Parameters
        ----------
        working_dir: str :


        Returns
        -------


        """
        if self._working_dir is not None:
            print(
                f"LogServer: {ANSI.WARNING}WARNING{ANSI.ENDC} > setWorkingdir() has been called more "
                "than one time!"
            )
            return
        self._working_dir = working_dir
        self._outputfile = open(
            f"{working_dir}/logs_" + time.strftime("%Y-%m-%d-%H-%M-%S"), "w"
        )

    def setIndent(self, indent: int = 0) -> int:
        self._indent_level = max([0, indent])
        return self._indent_level

    def indent(self, indent: int = 1) -> int:
        self._indent_level += indent
        return self._indent_level

    def unindent(self, indent: int = 1) -> int:
        self._indent_level -= indent
        self._indent_level = max([0, self._indent_level])
        return self._indent_level

    def printAndLog(
        self,
        name: str,
        tmess: str,
        message: str,
        flush: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        name: str :

        tmess: str :

        message: str :

        flush: bool :
             (Default value = False)

        Returns
        -------


        """
        indent_str = self._indent_str if tmess == "DEBUG" else ""
        header = f"{name}: {tmess} > " + indent_str * self._indent_level
        full_indent = " " * len(header)
        to_print = header + ("\n%s" % (full_indent)).join(message.split("\n"))
        self._logged.append(to_print)
        header = header.replace(tmess, f"{getattr(ANSI, tmess)}{tmess}{ANSI.ENDC}")
        for highlight in self.highlights:
            header = header.replace(
                highlight, f"{ANSI.HIGHLIGHT}{highlight}{ANSI.ENDC}"
            )
        to_print = header + ("\n%s" % (full_indent)).join(message.split("\n"))

        print(to_print)

        if flush or len(self._logged) > self.flush_every:
            self.flush()

    def flush(self) -> None:
        """

        Parameters
        ----------
        close: bool :
             (Default value = False)

        Returns
        -------


        """
        try:
            stdout.flush()  # Flushing stdout
            self._outputfile.write("\n".join(self._logged) + "\n")
            self._outputfile.flush()  # Flushing log file
            self._logged = []
        except AttributeError:
            if self.cannot_save_error:
                print(
                    f"LogServer: {ANSI.ERROR}ERROR{ANSI.ENDC} > Cannot save to file, "
                    "no working directory provided"
                )


class Logger:
    """Class Logger:
        Client logger class, to use everywhere. It prints (with colors) and logs in a file. In debug mode,
        every line is logged. See LogServer.

    Parameters
    ----------
    - namelogger :
        name of the logger
    - namelogger :
        name of the logger
        - verbosity [0 - 3]
        - 0: just errors
        - 1: warnings too
        - 2: info / successes too
        - 3: debug

    Returns
    -------

    """

    _logserver: Optional[LogServer] = None

    def __init__(
        self, namelogger: str, verbosity: int = 3, working_dir: Optional[str] = None
    ) -> None:
        self._verbosity = verbosity
        if self.__class__._logserver is None:
            self.__class__._logserver = LogServer(working_dir)
        self.setName(namelogger)

    def setName(self, namelogger):
        words = namelogger.split(":")
        new_words = []
        for word in words:
            if len(word) > 25:
                new_words.append("".join([let for let in word if let.isupper()]))
            else:
                new_words.append(word)
        self._name = ":".join(new_words)

    def setWorkingDir(self, working_dir: str) -> None:
        """

        Parameters
        ----------
        working_dir: str :


        Returns
        -------


        """
        self._logserver.setWorkingDir(working_dir)

    def setHighlights(self, highlights: List[str]) -> None:
        self._logserver.setHighlights(highlights)

    def flush(self) -> None:
        """

        Parameters
        ----------

        Returns
        -------


        """
        self._logserver.flush()

    def setIndent(self, indent: int = 0) -> int:
        return self._logserver.setIndent(indent)

    def indent(self, indent: int = 1) -> int:
        return self._logserver.indent(indent)

    def unindent(self, indent: int = 1) -> int:
        return self._logserver.unindent(indent)

    def debug(
        self,
        message: str = "",
        error: Optional[Exception] = None,
    ) -> None:
        """

        Parameters
        ----------
        message: str :
             (Default value = "")
        error: Exception :
             (Default value = None)

        Returns
        -------


        """
        if message == "" and error is not None:
            message = " | ".join([arg for arg in error.args if isinstance(arg, str)])
        if self._verbosity > 2:
            self._logserver.printAndLog(self._name, "DEBUG", message, flush=True)

    def success(
        self,
        message: str,
        flush: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        message: str :

        flush: bool :
             (Default value = False)

        Returns
        -------


        """
        if self._verbosity > 1:
            self._logserver.printAndLog(
                self._name,
                "SUCCESS",
                message,
                flush,
            )

    def info(
        self,
        message: str,
        flush: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        message: str :

        flush: bool :
             (Default value = False)

        Returns
        -------


        """
        if self._verbosity > 1:
            self._logserver.printAndLog(
                self._name,
                "INFO",
                message,
                flush,
            )

    def warn(
        self,
        message: str,
        flush: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        message: str :

        flush: bool :
             (Default value = False)

        Returns
        -------


        """
        if self._verbosity > 0:
            self._logserver.printAndLog(self._name, "WARNING", message, flush)

    def error(
        self,
        error: Exception,
    ) -> None:
        """

        Parameters
        ----------
        error: Exception :


        Returns
        -------


        """
        message = " | ".join([arg for arg in error.args if isinstance(arg, str)])
        self._logserver.printAndLog(self._name, "ERROR", message)
        self.flush()
        raise error

    # ASSERT UTILS

    def maybeError(self, error, debug):
        if debug:
            self.debug(error=error)
        else:
            self.error(error)

    def Assert(
        self,
        cond: bool,
        error: Exception,
        debug: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        cond: bool :

        error: Exception :

        debug: bool :
             (Default value = False)

        Returns
        -------


        """
        if not cond:
            self.maybeError(error, debug)

    def assertIsInstance(
        self,
        obj: object,
        name: str,
        _type: Union[type, Tuple[type], TypingType],
        length: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """

        Parameters
        ----------

        Returns
        -------


        """
        if not (ret := isInstance(obj, name, _type))[0]:
            name, _type = ret[1:]
            self.maybeError(
                TypeError(
                    f"{name} of type {printType(_type, embedded=True)} should be of type {_type}"
                ),
                debug=debug,
            )

        if length is not None:
            if hasattr(obj, "__len__"):
                self.Assert(
                    len(obj) == length,
                    ValueError(
                        f"{name} of length {len(obj)} should have length {length}"
                    ),
                )
            else:
                self.error(TypeError(f"{name} has no defined length"))

    def assertIs(
        self,
        obj: object,
        name: str,
        what: object,
        debug: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        obj: object :

        name: str :

        what: object :

        debug: bool :
             (Default value = False)

        Returns
        -------


        """
        self.Assert(
            obj is what,
            TypeError(
                f"{self._name} > {name} of type {type(obj)} should not be {what} "
                "(this probably indicates that the default value should have been overwritten)"
            ),
            debug=debug,
        )

    def assertIsNot(
        self,
        obj: object,
        name: str,
        what: object,
        debug: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        obj: object :

        name: str :

        what: object :

        debug: bool :
             (Default value = False)

        Returns
        -------


        """
        self.Assert(
            obj is not what,
            TypeError(
                f"{self._name} > {name} of type {type(obj)} should not be {what} "
                "(this probably indicates that the default value should have been overwritten)"
            ),
            debug=debug,
        )

    def assertIsNotNone(self, obj: object, name: str, debug: bool = False) -> None:
        """

        Parameters
        ----------
        obj: object :

        name: str :

        debug: bool :
             (Default value = False)

        Returns
        -------


        """
        self.Assert(
            obj is not None,
            TypeError(
                f"{self._name} > {name} should not be None "
                "(this probably indicates that the default value should have been overwritten)"
            ),
            debug=debug,
        )

    def assertIsIn(
        self,
        obj: Union[object, List[object]],
        name: Union[str, List[str]],
        iterable: Union[tuple, list, dict],
        iterable_name: str,
        debug: bool = False,
    ):
        """

        Parameters
        ----------
        obj: Union[object :

        List[object]] :

        name: Union[str :

        List[str]] :

        iterable: Union[tuple :

        list :

        dict] :

        iterable_name: str :

        debug: bool :
             (Default value = False)

        Returns
        -------


        """
        if isinstance(obj, list):
            if not isinstance(name, list):
                self._logserver.printAndLog(
                    f"{self._name} > logger",
                    "ERROR",
                    "If obj is a list, then name must also be a list!",
                )
                raise ValueError(
                    f"{self._name} > logger:If obj is a list, then name must also be a list!"
                )
            for _obj, _name in zip(obj, name):
                self.assertIsIn(_obj, _name, iterable, iterable_name, debug)

        else:
            self.Assert(
                obj in iterable,
                KeyError(f"{self._name} > {name} should be in {iterable_name}"),
                debug=debug,
            )

    def assertIsNotIn(
        self,
        obj: Union[object, List[object]],
        name: Union[str, List[str]],
        iterable: Union[tuple, list, dict],
        iterable_name: str,
        debug: bool = False,
    ):
        """

        Parameters
        ----------


        Returns
        -------


        """
        if isinstance(obj, list):
            if not isinstance(name, list):
                self._logserver.printAndLog(
                    f"{self._name} > logger",
                    "ERROR",
                    "If obj is a list, then name must also be a list!",
                )
                raise ValueError("If obj is a list, then name must also be a list!")
            for _obj, _name in zip(obj, name):
                self.assertIsNotIn(_obj, _name, iterable, iterable_name, debug=debug)

        else:
            self.Assert(
                obj not in iterable,
                KeyError(f"{self._name} > {name} should NOT be in {iterable_name}"),
                debug=debug,
            )

    def assertShapesMatch(self, debug: bool = False, **shapes: List[int]) -> None:
        """

        Parameters
        ----------
        debug: bool :
             (Default value = False)
        **shapes: List[int] :


        Returns
        -------


        """
        name0 = list(shapes.keys())[0]
        shape0 = shapes.pop(name0)
        for name, shape in shapes.items():
            self.Assert(
                len(shape) == len(shape0),
                error=ShapesMatchError(name0, shape0, name, shape),
                debug=debug,
            )
            self.Assert(
                all([x == y for x, y in zip(shape, shape0)]),
                error=ShapesMatchError(name0, shape0, name, shape),
                debug=debug,
            )

    # others

    def descriptiveStatistics(self, data: Jnp_ndarray, name: str, **kwargs) -> None:
        """

        Parameters
        ----------
        data: Jnp_ndarray :

        name: str :

        **kwargs :


        Returns
        -------


        """
        kwargs["return_err"] = True
        stats, err = descriptiveStatistics(data, mod=jnp, **kwargs)
        mess = f"{name} shows statistics"
        if err:
            self.error(ValueError(f"{mess} \n{stats}"))
        else:
            self.info(f"{mess} \n{stats}")

    def printTable(
        self,
        rows: List[str],
        columns: List[str],
        data: np.ndarray,
        level: str = "info",
        title: str = "",
        **kwargs,
    ):
        print_f = getattr(self, level)

        print_f(
            title
            + ("\n" if len(title) else "")
            + printTable(rows, columns, data, **kwargs)
        )

    def descriptiveStatisticsTable(
        self,
        level: str = "info",
        title: str = "",
        functions: Tuple[str] = ("min", "max", "mean", "std"),
        names: Optional[Dict[str, str]] = None,
        math: ModuleType = np,
        max_len_unwrap: int = 5,
        **data,
    ):

        print_f = getattr(self, level)

        print_f(
            title
            + ("\n" if len(title) else "")
            + descriptiveStatisticsTable(
                functions=functions,
                names=names,
                math=math,
                max_len_unwrap=max_len_unwrap,
                **data,
            )
        )

    def checkVJP(
        self,
        f: Callable,
        args: Tuple[Jnp_ndarray],
        f_name: str = "unnamed_function",
        eps: Union[float, List[float]] = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        plot: bool = True,
        randomize: int = 0,
        warn_if_min_above: float = 0.1,
        fatal_if_min_above: float = 1,
    ):
        self.info(f"Checking gradients of {f_name} numerically...")
        # self.assertIsInstance(
        #     ret, "f_returned_value", Union[jnp_ndarray, Tuple[jnp_ndarray]]
        # )

        eps = toList(eps)
        if randomize > 0:
            t0 = time.time()
            f_res, err = checkVJP(f, args, eps=eps)
            dt = time.time() - t0

            randomize -= 1
            err = np.median(
                np.stack(
                    [err]
                    + [
                        checkVJP(f, args, eps=eps)[-1]
                        for _ in (
                            trange(randomize)
                            if (randomize - 1) * dt > 10
                            else range(randomize)
                        )
                    ]
                ),
                axis=0,
            )
        else:
            f_res, err = checkVJP(f, args, eps=eps)

        print()

        min_eps = min(eps)
        if min_eps > fatal_if_min_above:
            self.error(ValueError(f"{f_name}: min relative error {min_eps}"))
        elif min_eps > warn_if_min_above:
            self.warn(f"{f_name}: min relative error {min_eps}")

        if len(eps) > 0:
            fig, ax = plt.subplots()
            ax.loglog(eps, err, c="r")
            ax.grid()
            ax.set_xlabel("eps")
            ax.set_ylabel("relative error")
            fig.suptitle(f_name)
            # no access to io, can't save here
        else:
            fig = None
        return f_res, fig
