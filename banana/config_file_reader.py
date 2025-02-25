import numpy as np
import os
from typing import List


class ConfigFileReader:
    """ """

    _debug = False
    mandatory_sections = [
        "general",
        "logging",
        "strategy",
    ]
    subconfig_sections = ["general", "logging", "strategy", "variables", "modules"]

    def __init__(self, file: str, args: List[str]):
        try:
            self.f = open(file, "r")
        except FileNotFoundError:
            raise FileNotFoundError("Configuration file not found:", file)

        self._replacement_map = dict(os.environ)
        for arg in args:
            w = arg.find("=")
            if w > 0:
                self._replacement_map[arg[:w]] = arg[w + 1 :]
            else:
                print(
                    f"WARNING: argument {arg} was not understood as a configuration argument"
                )

    def debug(self, message):
        """

        Parameters
        ----------
        message :


        Returns
        -------


        """
        if self._debug:
            print(f"DEBUG: {message}")

    def read(self):
        """ """
        lines = [line.replace("\n", "") for line in self.f.readlines()]
        config, section, sec_config = {}, "", {}
        for i, line in enumerate(lines):
            self.debug(f"Line {i}: Before stripping: {line}")
            comment = line.find("#")
            until = comment if comment >= 0 else len(line)
            line = line[:until].strip()
            self.debug(f"Line {i}: After stripping: {line}")
            if line == "":  # This is an empty line
                self.debug("Empty line")
                continue
            elif line[0] == "[]"[0]:  # This is a new section
                if section != "":
                    config[section] = sec_config
                section = line.replace("[", "").replace("]", "")
                self.debug(f"Found section {section}")
                sec_config = {}
            elif "+=" in line:  # We're updating a defined value
                w = line.find("+=")
                key = line[:w].strip()
                value = line[w + 2 :].strip()
                if key not in sec_config:
                    raise ValueError(
                        f"Line {i}: {section}>{key} not defined yet, "
                        f"did you mean to use '=' instead of '+='?"
                    )
                tval = self._interpretValue(self._interpretExternalVariables(value))
                if not isinstance(tval, type(sec_config[key])):
                    raise ValueError(
                        f"Line {i}: {section}>{key} cannot add types "
                        f"{type(sec_config[key])} and {type(tval)}",
                    )
                else:
                    # if this is a long string with spaces in it, add a space to replace new line
                    # elif sec_config is a list and tval is just an element of the list, append it
                    #      this is in case we forget the comma at the end of a line!
                    # else, directly apply '+=' operator
                    if isinstance(tval, str) and " " in tval or " " in sec_config[key]:
                        sec_config[key] += " " + tval
                    elif isinstance(sec_config[key], list) and isinstance(
                        tval, type(sec_config[key][-1])
                    ):
                        self.warn(
                            f"You forgot a comma after '{key} += {value}', "
                            "I got it right this time but please correct it! "
                        )
                        sec_config[key].append(tval)
                    else:
                        sec_config[key] += tval
            elif "=" in line:  # We're creating a new value
                w = line.find("=")
                key = line[:w].strip()
                value = line[w + 1 :].strip()
                tval = self._interpretValue(self._interpretExternalVariables(value))
                self.debug(
                    f"Found = operator with key {key} and value {value} of type {type(tval)}"
                )
                if key in sec_config:
                    raise ValueError(
                        f"Line {i}: {section}>{key} already defined, "
                        f"did you mean to use '+=' instead of '='?"
                    )
                sec_config[key] = tval
            else:  # WTF
                raise ValueError(f"Line {i}: Not understood\n> {line}")
        # We want to keep the last section!
        config[section] = sec_config

        for _sec in self.mandatory_sections:
            assert (
                _sec in config
            ), f"Section {_sec} is missing in the configuration file"
        for section, sec_config in config.items():
            self._dictifyKeys(sec_config)
            config[section] = getattr(self, "_postProcess" + section.capitalize())(
                sec_config
            )

        class SubConfig:
            """ """

            def __init__(self, subconfig):
                for k, v in subconfig.items():
                    self.__dict__[k] = v

        class Config:
            """ """

            def __init__(self, config, sc):
                for k, v in config.items():
                    if k in sc:
                        self.__dict__[k] = SubConfig(v)
                    else:
                        self.__dict__[k] = v

        return Config(config, self.subconfig_sections)

    def _dictifyKeys(self, config):
        """

        Parameters
        ----------
        config :


        Returns
        -------


        """
        key2del = []
        for key in list(config):
            value = config[key]
            if "." in key:
                w = key.find(".")
                key0 = key[:w]
                if key0 in config:
                    config[key0].update({key[w + 1 :]: value})
                else:
                    config[key0] = {key[w + 1 :]: value}
                key2del.append(key)
                self._dictifyKeys(config[key0])

        for key in key2del:
            del config[key]

    def _interpretExternalVariables(self, word):
        """
        Look for external variables in `word` and tried to replace them with values.

        An external variable has the form $VARIABLE in capital letter for instance.

        External variables are read from the environment variables or from sys.args, provided they
        are given with the form "VARIABLE=VALUE"

        This happens before type casting, commas in $VARIABLE will create lists and
        integers/float/etc will be interpreted as such.

        There can be more than one external value per word.

        Parameters
        ----------
        word: str:
            word to be parsed and possibly updated


        Returns
        -------
        word: str:
            updated word


        """
        if not ("$" in word):
            return word

        beg = word.find("$")
        end = beg
        for i in range(beg + 1, len(word)):
            if word[i].isupper() or word[i].isnumeric():
                end += 1
            else:
                break
        end += 1

        pattern = word[beg + 1 : end]
        try:
            replacement = self._replacement_map[pattern]
        except KeyError:
            to_print = "\n".join(
                [f"{k} = {v}" for k, v in self._replacement_map.items()]
            )
            raise ValueError(
                f"Cannot replace {pattern}. Replacement map is \n{to_print}"
            )
        if self.debug:
            print(f"DEBUG: replacing ${pattern} with {replacement}")
        word = word[:beg] + replacement + word[end:]
        return self._interpretExternalVariables(word)

    def _interpretValue(self, value):
        """

        Parameters
        ----------
        value :


        Returns
        -------


        """
        if value == "None":
            return None
        if value == "True":
            return True
        if value == "False":
            return False
        if "," in value:
            tval = []
            for val in value.split(","):
                tval.append(self._interpretValue(val.strip()))
            iv2d = [i for i, val in enumerate(tval) if val == ""]
            for i in iv2d[::-1]:
                tval.pop(i)
            return tval
        if "inf" in value:
            return float(value)
        if "." in value:
            try:
                return float(value)
            except ValueError:
                pass
        try:
            return int(value)
        except ValueError:
            return value

    def _assert(self, dic, namedict, key, tvalue):
        """

        Parameters
        ----------
        dic :

        namedict :

        key :

        tvalue :


        Returns
        -------


        """
        assert key in dic, f"{key} = [{tvalue}] should be provided in {namedict}"
        assert isinstance(
            dic[key], tvalue
        ), f"{namedict}:{key} expected type(s) {tvalue} but got {type(dic[key])}"

    def _postProcessGeneral(self, config):
        """

        Parameters
        ----------
        config :


        Returns
        -------


        """
        self._assert(config, "general", "description", str)
        self._assert(config, "general", "working_dir", str)
        self._assert(config, "general", "libraries_dirs", (type(None), list))
        return config

    def _postProcessLogging(self, config):
        """

        Parameters
        ----------
        config :


        Returns
        -------


        """
        self._assert(config, "logging", "verbosity_banana", int)
        self._assert(config, "logging", "debug_classes", (str, list))
        self._assert(config, "logging", "highlights", (type(None), list))
        config["highlights"] = (
            [] if config["highlights"] is None else config["highlights"]
        )
        self._assert(config, "logging", "run_debug", (bool))
        return config

    def _postProcessStrategy(self, config):
        """

        Parameters
        ----------
        config :


        Returns
        -------


        """
        self._assert(config, "strategy", "strategy", str)
        self._assert(config, "strategy", "args", dict)
        return config

    def _postProcessParameters(self, config):
        """

        Parameters
        ----------
        config :


        Returns
        -------


        """
        return config

    def _readOneFile(self, key, config):
        """

        Parameters
        ----------
        key :

        config :


        Returns
        -------


        """
        self._assert(config, f"observations.{key}", "file", str)
        filepath = config["file"]

        if filepath[-3:] == "npz":
            data = dict(np.load(filepath))
        elif filepath[-3:] == "npy":
            data = {key: np.load(filepath)}
        elif filepath[-3:] == "bin":
            self._assert(config, f"observations.{key}", "format", str)
            fmt = config["format"]
            data = {key: np.fromfile(filepath, fmt)}
        else:
            # else we assume file is text readable (csv, txt, ascii, etc)

        if "filters" in config:
            mask = np.zeros(list(data.values())[0].shape[-1], bool)
            for attr, val in config["filters"].items():
                self._assert(
                    config["filters"], f"observations.{key}.filters", attr, list
                )
                assert len(val) // 3 == len(val) / 3, (
                    f"Filter argument {attr} must be a list of triplets"
                    " key, min ,max"
                )
                _mask = np.ones_like(mask)
                for i in range(len(val) // 3):
                    key, _min, _max = val[i * 3], val[i * 3 + 1], val[i * 3 + 2]
                    self._assert(data, "observations file", key, np.ndarray)
                    _mask *= (data[key] > _min) * (data[key] < _max)
                    self.debug(
                        f"Croping key {key} with min/max "
                        f"{np.min(data[key])}/{np.max(data[key])} "
                        f"to range {_min}/{_max}"
                    )
                mask += _mask

            self.debug(f"Mask has size {mask.size} and {np.sum(mask)} selected entries")

            for key, val in data.items():
                data[key] = val[..., mask]
            data["mask"] = mask

        if "prefix" in config:
            return {f"{config['prefix']}_{k}": v for k, v in data.items()}
        else:
            return data

    def _postProcessObservations(self, config):
        """

        Parameters
        ----------
        config :


        Returns
        -------


        """
        ret = {}
        for key, val in config.items():
            self._assert(config, "observations", key, dict)
            new = self._readOneFile(key, val)
            for k in new:
                if k in ret:
                    raise ValueError(
                        f"Key {k} from observation file {key} is already defined"
                    )
            ret.update(new)
        return ret

    def _postProcessVariables(self, config):
        """

        Parameters
        ----------
        config :


        Returns
        -------


        """
        self._assert(config, "variables", "variables", list)
        return config

    def _postProcessModules(self, config):
        """

        Parameters
        ----------
        config :


        Returns
        -------


        """
        self._assert(config, "modules", "modules", list)
        return config
