from swarmfish.swarm_control import SwarmParams
from swarmfish.obstacles import PolygonObstacle
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
            log.info('Parsing json config')
            json_string = f.read()
            parsed = json.loads(json_string)
            return SwarmParams(**parsed['Agent']['SwarmParams'])
        except Exception as e:
            log.error(f'Parsing error {e}')
        finally:
            f.close()
    return None

def load_polygons_from_json(file_name: str) -> list[PolygonObstacle]:
    ''' load a list of polygons obstacles from json file
        generated with scenebuilder
    '''
    try:
        f = open(file_name)
    except FileNotFoundError:
        log.error(f'File {file_name} not found')
    else:
        try:
            log.info('Parsing json scene')
            json_string = f.read()
            parsed = json.loads(json_string)
            buildings = parsed['scenebuilder']['buildings']
            obstacles = []
            for building in buildings:
                name = building['ID']
                vertices = np.array(building['vertices'])
                obstacle = PolygonObstacle(name=name,
                        vertices=vertices[:,0:2],
                        z_min=0.,
                        z_max=vertices[0][2])
                obstacles.append(obstacle)
            return obstacles
        except Exception as e:
            log.error(f'Parsing error {e}')
        finally:
            f.close()
    return []
