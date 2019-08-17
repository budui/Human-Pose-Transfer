#!/usr/bin/env python3

"""
run.py is used to run engine.
In fact we can run each implementation(engine) separately,
but run each implementation in a unified way can avoid some duplicate code,
such as prepare gpu, read and save config file, and so on.
"""
import collections
from argparse import ArgumentParser
from importlib import import_module
from os import path, makedirs
from pprint import pprint

import toml
import torch

# python 3.8+ compatibility
try:
    collectionsAbc = collections.abc
except:
    collectionsAbc = collections


def update_config(config, config_update):
    for k, v in config_update.items():
        dv = config.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            config[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            config[k] = update_config(dv, v)
        else:
            config[k] = v
    return config


# each line represent a engine: key is name, value is the import path for this engine.
IMPLEMENTED_ENGINE = {
    "PG2-1": "implementations.PG2.train1",
    "PG2-2": "implementations.PG2.train2",
    "PG2-Generator": "generate"
}


def parse_argument():
    parser = ArgumentParser("Train")
    parser.add_argument("implementation", type=str, choices=IMPLEMENTED_ENGINE.keys(), help="run which?")
    parser.add_argument("-g", '--gpu_id', default=0, type=int, help='gpu_id: e.g. 0', required=True)
    parser.add_argument("-c", "--config", type=str, help="config file path", required=True)
    parser.add_argument("-o", "--output", type=str, help="output path", required=True)
    parser.add_argument("-t", "--toml", action="append", type=str, help="overwrite toml config use cli arg")
    options = parser.parse_args()
    return options


def prepare_gpu(gpu_ids):
    torch.cuda.set_device(gpu_ids)
    torch.backends.cudnn.benchmark = True


def load_config(config_path, overwrite_tomls):
    print("reading config from <{}>\n".format(path.abspath(config_path)))
    try:
        with open(config_path, "r") as f:
            config = toml.load(f)
            if overwrite_tomls is not None:
                config_update = toml.loads("\n".join(overwrite_tomls))
                print(config_update)
                config = update_config(config, config_update)
            return config
    except FileNotFoundError as e:
        print("can not find config file")
        raise e


def save_config(config, output_folder):
    if not path.exists(output_folder):
        makedirs(output_folder)
    with open(path.join(output_folder, "train.toml"), "w") as f:
        toml.dump(config, f)


def main():
    options = parse_argument()

    prepare_gpu(options.gpu_id)

    config = load_config(options.config, options.toml)
    config["output"] = options.output
    pprint(config)
    save_config(config, config["output"])

    engine = import_module(IMPLEMENTED_ENGINE[options.implementation])
    if IMPLEMENTED_ENGINE[options.implementation] == "generate":
        config["engine"] = options.implementation
    else:
        save_config(config, config["output"])

    print("#" * 80, "\n")

    engine.run(config)


if __name__ == '__main__':
    main()
