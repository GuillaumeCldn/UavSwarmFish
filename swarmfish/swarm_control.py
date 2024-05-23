import numpy as np
import math
from dataclasses import dataclass

@dataclass
class SwarmParams:
    # collective motion parameters
    max_velocity: float
    min_velocity: float
    velocity: float
    alt: float
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



def wrap_to_pi(angle) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

@dataclass
class State:
    pos: np.ndarray
    speed: np.ndarray
    heading: float
    timestamp: float

    #TODO getter functions for velocity, distance, heading diff, etc
    
    def get_distance_2d(self, other: 'State') -> float:
        return np.linalg.norm(self.pos[0:2] - other.pos[0:2])

    def get_distance_3d(self, other: 'State') -> float:
        return np.linalg.norm(self.pos - other.pos)

    def get_distance_coupled(self, other: 'State', params: SwarmParams) -> float:
        return np.sqrt(np.sum((self.pos[0:2] - other.pos[0:2])**2) + params.alpha_z * (self.pos[2] - other.pos[2])**2)

    def get_speed_2d(self) -> float:
        return np.linalg.norm(self.speed[0:2])

    def get_speed_3d(self) -> float:
        return np.linalg.norm(self.speed)

    def get_vz(self) -> float:
        return self.speed[2]

    def get_course(self, use_heading: bool = False) -> float:
        if use_heading:
            return self.heading
        return math.atan2(self.speed[1], self.speed[0])

    def get_course_diff(self, other: 'State', use_heading: bool = False) -> float:
        diff = other.get_course(use_heading) - self.get_course(use_heading)
        return wrap_to_pi(diff)

    def get_viewing_angle(self, other: 'State', use_heading: bool = False) -> float:
        ''' Viewing angle in 2D
        '''
        diff = other.pos[0:2] - self.pos[0:2]
        direction = math.atan2(diff[1], diff[0])
        viewing_angle = direction - self.get_course(use_heading)
        return wrap_to_pi(viewing_angle)

    def __str__(self):
        out = f'State: '
        out += f'pos= {self.pos[0]:.2f}, {self.pos[1]:.2f}, {self.pos[2]:.2f} | '
        out += f'speed= {self.speed[0]:.2f}, {self.speed[1]:.2f}, {self.speed[2]:.2f} | '
        out += f'vel= {self.get_speed_3d():.2f} course= {np.degrees(self.get_course()):0.2f} '
        out += f'heading= {np.degrees(self.heading):.2f} | '
        out += f'time= {self.timestamp:.3f}'
        return out

@dataclass
class Commands:
    delta_heading: float
    delta_speed: float
    delta_vz: float


##
## interaction functions
##

def interaction_wall(agent: State, params: SwarmParams, wall: (float, float) = None, z_min: float = None, z_max: float = None) -> tuple[float, float]:
    ''' Interaction with a wall defined by a distance and relative orientation
        and with floor and ceiling
    '''
    dyaw = 0.
    dvz = 0.
    if wall is not None:
        dist, angle = wall
        fw = math.exp(-(dist / params.lw)**2)
        ow = params.ew1 * math.cos(angle) + params.ew2 * math.cos(2. * angle)
        dyaw = params.yw * math.sin(angle) * (1. + ow) * fw
    if z_min is not None:
        dz = agent.pos[2] - z_min
        dvz += 2. * params.y_perp / (1. + math.exp((dz - params.dz0) / params.dz0))
    if z_max is not None:
        dz = z_max - agent.pos[2]
        dvz += 2. * params.y_perp / (1. + math.exp((dz - params.dz0) / params.dz0))
    #TODO a speed variation to slow down on obstacles ?
    return dyaw, dvz

def interaction_social(agent: State, params: SwarmParams, other: State, r_w: float = 100.) -> tuple[float, float, float, float]:
    ''' Social interaction of alignment, attraction, speed and vertical speed
        with wall distance r_w for attenuation
    '''
    attenuation = 1. - math.exp(-(r_w/params.lw)**2)        # wall attenuation
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
    return dyaw_ali + dyaw_att, dv, dvz

def interaction_nav(agent: State, params: SwarmParams, direction: float = None, altitude: float = None) -> tuple[float, float]:
    ''' Navigation interaction with a specified direction and altitude
        also add the vertical speed damping
    '''
    dyaw = 0.
    dvz = 0.
    if direction is not None and agent.get_speed_2d() > 0.5:
        dyaw = params.y_nav * math.sin(direction - agent.get_course(params.use_heading))
    if altitude is not None:
        dvz += -params.y_perp * math.tanh((agent.pos[2] - altitude) / params.a_z)
    # add vertical speed damping
    speed = agent.get_speed_3d()
    if speed > 0.1:
        dvz += - params.y_para * agent.get_vz() / speed
    #TODO speed attraction to setpoint
    return dyaw, dvz

def interaction_intruder(agent: State, params: SwarmParams, other: State) -> tuple[float, float]:
    ''' Interaction of repulsion with an intruder
    '''
    dij = agent.get_distance_coupled(other, params)
    dphi = agent.get_course_diff(other, params.use_heading)
    psi = agent.get_viewing_angle(other, params.use_heading)
    #FIXME use dphi or psi for even function ?
    dyaw = -params.y_intruder * math.exp((dij / params.l_intruder)**2) * (1. + params.ew1 * math.cos(psi)) * math.sin(dphi)
    dz = agent.pos[2] - other.pos[2]
    dvz = params.y_z_intruder * math.tanh(dz / params.a_z) * math.exp(-(dij / params.L_z_2)**2) #FIXME name of L_z_2
    return dyaw, dvz

def compute_interactions(
        agent: State,
        params: SwarmParams,
        neighbors: list[State] = [],
        nb_influent: int = 1,
        direction: float = None,
        altitude: float = None,
        wall: (float, float) = None,
        z_min: float = None,
        z_max: float = None,
        intruders: list[State] = []) -> tuple[float, float]:
    # social
    social = []
    for neighbor in neighbors:
        social.append(interaction_social(agent, params, neighbor)) # FIXME compute r_w
    influential = sorted(social, key=lambda s: s[0])
    dyaw_s, dspeed_s, dvz_s = 0., 0., 0.
    for i, s in enumerate(influential):
        if i == nb_influent:
            #print("social",i,influential,social)
            break
        dyaw_s += s[0]
        dspeed_s += s[1]
        dvz_s += s[2]
    # wall and borders
    dyaw_w, dvz_w = interaction_wall(agent, params, wall, z_min, z_max)
    # navigation
    dyaw_nav, dvz_nav = interaction_nav(agent, params, direction, altitude)
    # intruders
    dyaw_i, dvz_i = 0., 0.
    for intruder in intruders:
        dy, dvz = interaction_intruder(agent, params, intruder)
        dyaw_i += dy
        dvz_i += dvz

    delta_yaw = dyaw_s + dyaw_w + dyaw_nav + dyaw_i
    print(f'  delta_yaw {np.degrees(delta_yaw):.2f} | s={np.degrees(dyaw_s):.2f}, w={np.degrees(dyaw_w):.2f}, n={np.degrees(dyaw_nav):.2f}, i={np.degrees(dyaw_i):.2f}')
    delta_speed = dspeed_s
    delta_vz = dvz_s + dvz_w + dvz_nav + dvz_i
    print(f'  delta_vz {delta_vz:.2f} | s={dvz_s:.2f}, w={dvz_w:.2f}, n={dvz_nav:.2f}, i={dvz_i:.2f}')
    return np.array([delta_yaw, delta_speed, delta_vz])


# TODO make a better obstacle class
class Arena:
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


# TODO move somewhere else, not needed here
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

