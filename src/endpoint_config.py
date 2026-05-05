import json
from pathlib import Path

FILE_PATH = Path(__file__)
CONFIG_PATH = FILE_PATH.parent / "endpoints/configs"


def endpoint_resolver(config, endpoint):
    if config != None:
        config = json.load(open(config))
    else:
        try:
            config = json.loads((CONFIG_PATH / f"{endpoint.replace('-','_')}.json").read_text())
        except FileNotFoundError as e:
            raise ValueError("Endpoint not supported") from e
    return config
