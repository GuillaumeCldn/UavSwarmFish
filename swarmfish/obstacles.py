from swarmfish.swarm_control import State, SwarmParams, wrap_to_pi
import numpy as np
import math

class Obstacle:

    def get_wall(self, agent: State, params: SwarmParams) -> tuple[float, float]:
        pass


class Arena(Obstacle):
    center: np.ndarray
    radius: float

    def __init__(self, center: np.ndarray = None, radius: float = 100.):
        if center is None:
            center = np.array([0., 0.])
        self.center = center
        self.radius = radius

    def get_wall(self, agent: State, params: SwarmParams):
        dpos = agent.pos[0:2] - self.center
        dist = self.radius - np.linalg.norm(dpos)
        course = agent.get_course(params.use_heading)
        theta = math.atan2(dpos[1], dpos[0])
        angle = course - theta
        return dist, wrap_to_pi(angle)

class CircleObstacle(Obstacle):
    center: np.ndarray
    radius: float
    z_min: float = None
    z_max: float = None

    def __init__(self, center: np.ndarray = None, radius: float = 100., z_min: float = None, z_max: float = None):
        if center is None:
            center = np.array([0., 0.])
        self.center = center
        self.radius = radius
        self.z_min = z_min
        self.z_max = z_max

    def get_wall(self, agent: State, params: SwarmParams):
        if self.z_min is not None and agent.pos[2] < self.z_min:
            return None
        if self.z_max is not None and agent.pos[2] > self.z_max:
            return None
        dpos = agent.pos[0:2] - self.center
        dist = np.linalg.norm(dpos) - self.radius
        course = agent.get_course(params.use_heading)
        theta = math.atan2(dpos[1], dpos[0])
        angle = course - theta
        return dist, wrap_to_pi(angle)

