import argparse
import importlib.util
import json
import argparse
import importlib.util

def load_dict(path):
    spec = importlib.util.spec_from_file_location("config", path)
    spec_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spec_module)
    spec = spec_module
    return spec.config

def parse_bool(value):
    if value.lower() in ['true']:
        return True
    elif value.lower() in ['false']:
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {value}')

def parse_list(value):
    try:
        parsed_value = eval(value)
        if not isinstance(parsed_value, list):
            raise argparse.ArgumentTypeError(f'List value expected, got {value}')
        return parsed_value
    except:
        raise argparse.ArgumentTypeError(f'List value expected, got {value}')

def add_arguments(parser, config, prefix=''):
    for key, value in config.items():
        new_prefix = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            add_arguments(parser, value, new_prefix)
        else:
            arg_name = f'-{new_prefix}'
            value_type = type(value)
            if value_type == bool:
                parser.add_argument(arg_name, type=parse_bool, default=value, help=f'{arg_name} value')
            elif value_type == list:
                parser.add_argument(arg_name, type=parse_list, default=value, help=f'{arg_name} value')
            else:
                parser.add_argument(arg_name, type=value_type, default=value, help=f'{arg_name} value')

def update_config(args, config, prefix=''):
    for key, value in config.items():
        new_prefix = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            update_config(args, value, new_prefix)
        else:
            config[key] = getattr(args, new_prefix)

def load_config(dict_path=None):
    path_parser = argparse.ArgumentParser(add_help=False)
    path_parser.add_argument('-config', type=str, default=dict_path, help='Path to the dictionary file')
    path_args, unknown_args = path_parser.parse_known_args()

    assert path_args.config is not None, 'Config Path Not Provided'

    config = load_dict(path_args.config)

    parser = argparse.ArgumentParser(description='Update config dictionary based on command line arguments.')

    add_arguments(parser, config)
    args = parser.parse_args(unknown_args)

    update_config(args, config)

    return config
