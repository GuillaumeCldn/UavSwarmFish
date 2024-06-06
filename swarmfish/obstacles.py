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
        dpos = self.center - agent.pos[0:2]
        dist = np.linalg.norm(dpos) - self.radius
        course = agent.get_course(params.use_heading)
        theta = math.atan2(dpos[1], dpos[0])
        angle = course - theta
        return dist, wrap_to_pi(angle)


class PolygonObstacle(Obstacle):
    '''
    2D polygon with min and max alt
    Vertices are [2 x N] array with x and y position of N vertices
    '''
    vertices: np.ndarray
    nb_vert: int = 0
    z_min: float = None
    z_max: float = None

    def __init__(self, vertices: np.ndarray, z_min: float = None, z_max: float = None):
        self.vertices = vertices
        self.z_min = z_min
        self.z_max = z_max
        self.nb_vert = np.shape(vertices)[1] # numbers of columns
        if self.nb_vert < 4:
            raise Exception('polygon obstacle too small')

    def get_wall(self, agent: State, params: SwarmParams = None):
        if self.z_min is not None and agent.pos[2] < self.z_min:
            return None
        if self.z_max is not None and agent.pos[2] > self.z_max:
            return None
        if params is None:
            use_heading = False
        else:
            use_heading = params.use_heading

        # find closest
        distances = np.array([ np.linalg.norm(self.vertices[:,i] - agent.pos[0:2]) for i in range(self.nb_vert) ])
        idx = np.argmin(distances)

        closest_pt = self.vertices[:,idx]
        next_vect = self.vertices[:, (idx+1) % self.nb_vert] - closest_pt
        prev_vect = self.vertices[:, (idx-1) % self.nb_vert] - closest_pt
        next_vect /= np.linalg.norm(next_vect)
        prev_vect /= np.linalg.norm(prev_vect)
        dpos = agent.pos[0:2] - closest_pt
        scal_next = np.dot(next_vect, dpos)
        scal_prev = np.dot(prev_vect, dpos)
        if scal_next > 0.:
            proj = closest_pt + scal_next * next_vect / np.linalg.norm(next_vect)
        elif scal_prev > 0.:
            proj = closest_pt + scal_prev * prev_vect / np.linalg.norm(prev_vect)
        else:
            proj = closest_pt
        dpos = proj - agent.pos[0:2]
        dist = np.linalg.norm(dpos)
        course = agent.get_course(use_heading)
        theta = math.atan2(-dpos[1], -dpos[0])
        angle = course - theta
        return dist, wrap_to_pi(angle)



def test():
    # test polygon
    vertices = np.array([
        [3., 3., -1., -1.],
        [0., 5., 5., 1.]
        ])

    polygon = PolygonObstacle(vertices)

    agent =  State(np.array([5., 1., 0.]), np.array([0., 1., 0.]), 0., 0.)
    print(agent)
    dist, angle = polygon.get_wall(agent)
    print(f'dist = {dist:.2f}, angle = {np.degrees(angle):.2f}')

    agent =  State(np.array([1., -1., 0.]), np.array([0., 1., 0.]), 0., 0.)
    print(agent)
    dist, angle = polygon.get_wall(agent)
    print(f'dist = {dist:.2f}, angle = {np.degrees(angle):.2f}')

    agent =  State(np.array([8., -5, 0.]), np.array([1., 1., 0.]), 0., 0.)
    print(agent)
    dist, angle = polygon.get_wall(agent)
    print(f'dist = {dist:.2f}, angle = {np.degrees(angle):.2f}')

    agent =  State(np.array([0., 10., 0.]), np.array([1., 0., 0.]), 0., 0.)
    print(agent)
    dist, angle = polygon.get_wall(agent)
    print(f'dist = {dist:.2f}, angle = {np.degrees(angle):.2f}')

if __name__ == '__main__':
    test()

