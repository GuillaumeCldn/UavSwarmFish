import numpy as np
import math
from dataclasses import dataclass

@dataclass
class SwarmParams:
    # collective motion parameters
    body_length: float
    max_velocity: float
    min_velocity: float
    velocity: float
    mind2d: float # RM ?
    fluct: float # RM ?

    ew1: float
    ew2: float
    alpha: float
    yw: float
    lw: float
    yatt: float
    latt: float
    d0att: float
    yali: float
    lali: float
    d0ali: float
    alt: float
    walldistance: float
    yacc: float
    lacc: float
    dv0: float
    ew1_ob: float
    ew2_ob: float
    yob: float
    lob: float
    y_perp: float
    y_para: float
    y_z: float
    a_z: float
    zmax: float
    zmin: float
    dz0: float
    alpha_z: float
    L_z_2: float
    y_nav: float
    y_intruder: float
    l_intruder: float

    use_heading: bool = False



def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

@dataclass
class State:
    pos: np.ndarray
    speed: np.ndarray
    heading: float
    timestamp: float

    #TODO getter functions for velocity, distance, heading diff, etc
    
    def get_distance_2d(self, other: State):
        return np.linalg.norm(self.pos[0:2] - other.pos[0:2])

    def get_distance_3d(self, other: State):
        return np.linalg.norm(self.pos - other.pos)

    def get_distance_coupled(self, other: State, params: SwarmParams):
        return np.sqrt((self.pos[0:2] - other.pos[0:2])**2 + params.alpha_z * (self.pos[2] - other.pos[2])**2)

    def get_speed_2d(self):
        return np.linalg.norm(self.speed[0:2])

    def get_speed_3d(self):
        return np.linalg.norm(self.speed)

    def get_vz(self):
        return self.speed[2]

    def get_course(self, use_heading: bool = False):
        if use_heading:
            return self.heading
        return math.atan2(self.speed[1], self.speed[0])

    def get_course_diff(self, other: State, use_heading: bool = False):
        diff = other.get_course(use_heading) - self.get_course(use_heading)
        return wrap_to_pi(diff)

    def get_viewing_angle(self, other: np.ndarray, use_heading: bool = False):
        ''' Viewing angle in 2D
        '''
        diff = other[0:2] - self.pos[0:2]
        direction = math.atan2(diff[1], diff[0])
        viewing_angle = direction - self.get_course(use_heading)
        return wrap_to_pi(viewing_angle)


@dataclass
class Commands:
    delta_heading: float
    delta_speed: float
    delta_vz: float


##
## interaction functions
##

def interaction_wall(agent: State, params: SwarmParams, dist: float, angle: float):
    fw = math.exp(-(dist / params.lw)**2)
    ow = params.ew1 * math.cos(angle) + params.ew2 * math.cos(2. * angle)
    delta_heading = params.yw * math.sin(angle) * (1. + ow) * fw
    return delta_heading

def interaction_social(agent: State, params: SwarmParams, other: State, r_w: float = 100.):
    ''' Social interaction of alignment, attraction, speed and vertical speed
        with wall distance r_w for attenuation
    '''
    attenuation = 1. - math.exp(-(params.rw/params.lw)**2)  # wall attenuation
    dij = agent.get_distance_coupled(other, params)         # distance between agents
    # alignment
    dphi = agent.get_course_diff(other, params.use_heading) # course/heading difference
    dyaw_ali = params.yali * ((dij + params.d0ali) / params.d0ali) * math.exp(-(dij/params.lali)**2) * math.sin(dphi) * attenuation
    # attraction
    psi = agent.get_viewing_angle(other, params.use_heading) # viewing angle
    dyaw_att = params.yatt * ((dij / params.d0att - 1.) / (1. + (dij / params.latt)**2)) * math.sin(psi) * attenuation
    # speed
    #FIXME normalize dv0 - dij
    dv = params.yacc * math.cos(psi) * (params.dv0 - dij) / (1. + dij / params.lacc)
    # vertical
    #FIXME check sign in tanh, L_z_2 name
    dz = agent.pos[2] - other.pos[2]
    dvz = params.y_z * math.tanh((dz - params.dz0) / params.a_z) * math.exp(-(dij / params.L_z_2)**2)
    return dyaw_ali, dyaw_att, dv, dvz

def interaction_vertical_damping(agent: State, params: SwarmParams):
    speed = agent.get_speed()
    if speed > 0.1:
        dvz = - params.y_para * agent.get_vz() / agent.get_speed()
    else:
        dvz = 0
    return dvz

def interaction_nav(agent: State, params: SwarmParams, direction: float = None, altitude: float = None):
    dyaw = 0.
    dvz = 0.
    if direction is not None:
        dyaw = params.y_nav * math.sin(direction - agent.get_course(params.use_heading))
    if altitude is not None:
        dvz = -params.y_perp * math.tanh((agent.pos[2] - altitude) / params.a_z)
    #TODO speed attraction to setpoint
    return dyaw, dvz

def interaction_intruder(agent: State, params: SwarmParams, other: State):
    dij = agent.get_distance_coupled(other, params)
    dphi = agent.get_course_diff(other, params.use_heading)
    psi = agent.get_viewing_angle(other, params.use_heading)
    #FIXME use dphi or psi for even function ?
    dyaw = -params.y_intruder * math.exp((dij / params.l_intruder)**2) * (1. + params.ew1 * math.cos(psi)) * math.sin(dphi)
    dz = agent.pos[2] - other.pos[2]
    dvz = params.y_z_intruder * math.tanh(dz / params.a_z) * math.exp(-(dij / params.L_z_2)**2) #FIXME name of L_z_2
    return dyaw, dvz




class Agent:
    name: str
    agent: State
    neighbors = {}

    def __init__(self, name: str, agent: State):
        self.name = name
        self.agent = agent

    def update_agent(self, pos: np.ndarray, speed: np.ndarray, heading: float, timestamp: float):
        self.agent.pos = pos
        self.agent.speed = speed
        self.agent.heading = heading
        self.agent.timestamp = timestamp

    def update_neighbor(self, name: str, pos: np.ndarray, speed: np.ndarray, heading: float, timestamp: float):
        if name in self.neighbors:
            self.neighbors[name].pos = pos
            self.neighbors[name].speed = speed
            self.neighbors[name].heading = heading
            self.neighbors[name].timestamp = timestamp
        else:
            self.neighbors[name] = State(pos, speed, heading, timestamp)

    def __str__(self):
        nb = len(self.neighbors)
        return f'agent {self.name} at {self.agent}\nwith {nb} neighbors:\n{self.neighbors}\n'


if __name__ == '__main__':
    # run tests
    state = State(np.random.rand(3), np.zeros(3), 0., 0.)
    uav = Agent("uav0", state)
    for i in range(5):
        ns = State(np.random.rand(3), np.zeros(3), float(i+1), 0.)
        uav.update_neighbor('uav'+str(i+1), ns.pos, ns.speed, ns.heading, ns.timestamp)

    print('uav state', uav)

