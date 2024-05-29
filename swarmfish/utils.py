from swarmfish.swarm_control import SwarmParams
import yaml
import json
import logging
import numpy as np

log = logging.getLogger(__name__)

def load_params_from_yaml(file_name: str) -> SwarmParams | None:
    try:
        f = open(file_name)
    except FileNotFoundError:
        log.error(f'File {file_name} not found')
    else:
        try:
            log.info('Parsing yaml config')
            yaml_string = f.read()
            parsed = yaml.load(yaml_string, Loader=yaml.FullLoader)
            return SwarmParams(**parsed['Agent']['SwarmParams'])
        except Exception as e:
            log.error(f'Parsing error {e}')
        finally:
            f.close()
    return None

def load_params_from_json(file_name: str) -> SwarmParams | None:
    try:
        f = open(file_name)
    except FileNotFoundError:
        log.error(f'File {file_name} not found')
    else:
        try:
            log.info('Parsing yaml config')
            json_string = f.read()
            parsed = json.loads(json_string)
            return SwarmParams(**parsed['Agent']['SwarmParams'])
        except Exception as e:
            log.error(f'Parsing error {e}')
        finally:
            f.close()
    return None

