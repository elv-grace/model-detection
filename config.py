import yaml
import os
from typing import Any

def load_config() -> Any: # same as model-celeb/config.py and model-vector's
    path = os.getenv('CONFIG_PATH', 'config.yml')
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    # for all paths in storage, if they don't start with "/", they are relative, make them absolute
    for key in config.get('storage', {}):
        if not config['storage'][key].startswith('/'):
            config['storage'][key] = os.path.join(os.path.dirname(path), config['storage'][key])
    return config

config = load_config()
