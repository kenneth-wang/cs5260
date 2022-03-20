import yaml


def load_yaml_config(yaml_file_path):
    """ Load yaml config file into dictionary.
    """
    with open(yaml_file_path) as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    return yaml_dict


config_dict = load_yaml_config("./config/config.yaml")
custom_config_dict = load_yaml_config("./config/config_custom_data.yaml")

PATHS = config_dict["PATHS"]
PARAMS = config_dict["PARAMS"]
VARIABLES = config_dict["VARIABLES"]

CUSTOM_PATHS = custom_config_dict["PATHS"]
CUSTOM_PARAMS = custom_config_dict["PARAMS"]
CUSTOM_VARIABLES = custom_config_dict["VARIABLES"]