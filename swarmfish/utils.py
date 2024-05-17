from swarmfish.swarm_control import SwarmParams
import yaml
import json

def load_params_from_yaml(file_name: str) -> SwarmParams | None:
    with open(file_name) as f:
        yaml_string = f.read()
        parsed = yaml.load(yaml_string, Loader=yaml.FullLoader)
        print(parsed)
        return SwarmParams(**parsed['Agent'])
    return None

def load_params_from_json(file_name: str) -> SwarmParams | None:
    with open(file_name) as f:
        json_string = f.read()
        parsed = json.loads(json_string)
        return SwarmParams(**parsed['Agent'])
    return None

