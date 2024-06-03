import numpy as np
import math
from dataclasses import dataclass

@dataclass
class SwarmParams:
    # collective motion parameters
    max_velocity: float
    min_velocity: float
    zmax: float
    zmin: float

    # wall
    y_w: float
    l_w: float
    e_w1: float
    e_w2: float
    # attraction
    y_att: float
    l_att: float
    d0_att: float
    a_att: float
    b_att: float
    # alignment
    y_ali: float
    l_ali: float
    d0_ali: float
    a_ali: float
    b_ali: float
    # speed
    y_acc: float
    l_acc: float
    d0_v: float
    # vertical
    y_z: float
    l_z: float
    a_z: float
    d0_z: float
    sigma_z: float
    # nav
    y_nav: float
    y_perp: float
    dz_perp: float
    y_para: float
    # intruders
    y_intruder: float
    y_z_intruder: float
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

    def get_distance_2d(self, other: 'State') -> float:
        return np.linalg.norm(self.pos[0:2] - other.pos[0:2])

    def get_distance_3d(self, other: 'State') -> float:
        return np.linalg.norm(self.pos - other.pos)

    def get_distance_coupled(self, other: 'State', params: SwarmParams) -> float:
        return np.sqrt(np.sum((self.pos[0:2] - other.pos[0:2])**2) + ((self.pos[2] - other.pos[2]) / params.sigma_z)**2)

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
class SwarmCommands:
    delta_course: float = 0.
    delta_speed: float = 0.
    delta_vz: float = 0.

    def __add__(self, other):
        return SwarmCommands(
                delta_course = wrap_to_pi(self.delta_course + other.delta_course),
                delta_speed = self.delta_speed + other.delta_speed,
                delta_vz = self.delta_vz + other.delta_vz)

    def __iadd__(self, other):
        self.delta_course = wrap_to_pi(self.delta_course + other.delta_course)
        self.delta_speed += other.delta_speed
        self.delta_vz += other.delta_vz
        return SwarmCommands(self.delta_course, self.delta_speed, self.delta_vz)

    def __str__(self):
        out = f'Cmd: '
        out += f'd_course= {np.degrees(self.delta_course):0.2f} '
        out += f'd_speed= {self.delta_speed:0.2f} '
        out += f'd_vz= {self.delta_vz:0.2f}'
        return out


##
## interaction functions
##

def interaction_wall(agent: State, params: SwarmParams, wall: (float, float) = None, z_min: float = None, z_max: float = None) -> tuple[float, float]:
    ''' Interaction with a wall defined by a distance and relative orientation
        and with floor and ceiling
    '''
    cmd = SwarmCommands()
    if wall is not None:
        dist, angle = wall
        fw = math.exp(-(dist / params.l_w)**2)
        ow = params.e_w1 * math.cos(angle) + params.e_w2 * math.cos(2. * angle)
        cmd.delta_course = params.y_w * math.sin(angle) * (1. + ow) * fw
    if z_min is not None:
        dz = agent.pos[2] - z_min
        cmd.delta_vz += 2. * params.y_perp / (1. + math.exp((dz - params.dz_perp) / params.dz_perp))
    if z_max is not None:
        dz = z_max - agent.pos[2]
        cmd.delta_vz += 2. * params.y_perp / (1. + math.exp((dz - params.dz_perp) / params.dz_perp))
    #TODO a speed variation to slow down on obstacles ?
    return cmd

def interaction_social(agent: State, params: SwarmParams, other: State, r_w: float = 100.) -> tuple[float, float, float, float]:
    ''' Social interaction of alignment, attraction, speed and vertical speed
        with wall distance r_w for attenuation
    '''
    attenuation = 1. - math.exp(-(r_w/params.l_w)**2)        # wall attenuation
    dij = agent.get_distance_coupled(other, params)          # distance between agents
    dphi = agent.get_course_diff(other, params.use_heading)  # course/heading difference
    psi = agent.get_viewing_angle(other, params.use_heading) # viewing angle

    # alignment
    f_ali = params.y_ali * (dij / params.d0_ali + 1.) * math.exp(-(dij/params.l_ali)**2)
    o_ali = math.sin(dphi) * (1. + params.a_ali * math.cos(2. * dphi))
    e_ali = 1. + params.b_ali * math.cos(psi)
    dyaw_ali = f_ali * o_ali * e_ali * attenuation

    # attraction
    f_att = params.y_att * ((dij / params.d0_att - 1.) / (1. + (dij / params.l_att)**2))
    o_att = math.sin(psi) * (1. - params.a_att * math.cos(psi))
    e_att = 1. - params.b_att * math.cos(dphi)
    dyaw_att = f_att * o_att * e_att * attenuation
    #print(f'dyaw_att {np.degrees(dyaw_att):0.2f} | dyaw_ali {np.degrees(dyaw_ali):0.2f} | social {np.degrees(dyaw_att+dyaw_ali):.2f}')

    # speed
    dv = params.y_acc * math.cos(psi) * ((params.d0_v - dij) / params.d0_v) / (1. + dij / params.l_acc)

    # vertical
    dz = other.pos[2] - agent.pos[2]
    dvz = params.y_z * math.tanh((dz - math.copysign(params.d0_z, dz)) / params.a_z) * math.exp(-(dij / params.l_z)**2)
    return SwarmCommands(delta_course=dyaw_ali+dyaw_att, delta_speed=dv, delta_vz=dvz)

def interaction_nav(agent: State, params: SwarmParams, direction: float = None, altitude: float = None) -> tuple[float, float]:
    ''' Navigation interaction with a specified direction and altitude
        also add the vertical speed damping
    '''
    cmd = SwarmCommands()
    if direction is not None and agent.get_speed_2d() > 0.5:
        cmd.delta_course = params.y_nav * math.sin(direction - agent.get_course(params.use_heading))
    if altitude is not None:
        cmd.delta_vz += -params.y_perp * math.tanh((agent.pos[2] - altitude) / params.a_z)
    # add vertical speed damping
    speed = agent.get_speed_3d()
    if speed > 0.1:
        cmd.delta_vz += - params.y_para * agent.get_vz() / speed
    #TODO speed attraction to setpoint
    return cmd

def interaction_intruder(agent: State, params: SwarmParams, other: State) -> tuple[float, float]:
    ''' Interaction of repulsion with an intruder
    '''
    dij = agent.get_distance_coupled(other, params)
    dphi = agent.get_course_diff(other, params.use_heading)
    psi = agent.get_viewing_angle(other, params.use_heading)
    #FIXME use dphi or psi for even function ?
    dyaw = -params.y_intruder * math.exp(-(dij / params.l_intruder)**2) * (1. + params.e_w1 * math.cos(psi)) * math.sin(dphi)
    dz = agent.pos[2] - other.pos[2]
    dvz = params.y_z_intruder * math.tanh(dz / params.a_z) * math.exp(-(dij / params.l_z)**2)
    return SwarmCommands(delta_course=dyaw, delta_vz=dvz)

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
        social.append(interaction_social(agent, params, neighbor, r_w=wall[0]))
    influential = sorted(social, key=lambda s: np.fabs(s.delta_course), reverse=True)
    d_social = SwarmCommands()
    for i, s in enumerate(influential):
        if i == nb_influent:
            break
        d_social += s
    # wall and borders
    d_borders = interaction_wall(agent, params, wall, z_min, z_max)
    # navigation
    d_nav = interaction_nav(agent, params, direction, altitude)
    # intruders
    d_intruders = SwarmCommands()
    for intruder in intruders:
        d_intruders += interaction_intruder(agent, params, intruder)

    cmd = d_social + d_borders + d_nav + d_intruders
    #print(f'  delta_yaw {np.degrees(delta_yaw):.2f} | s={np.degrees(dyaw_s):.2f}, w={np.degrees(dyaw_w):.2f}, n={np.degrees(dyaw_nav):.2f}, i={np.degrees(dyaw_i):.2f}')
    #print(f'  delta_vz {delta_vz:.2f} | s={dvz_s:.2f}, w={dvz_w:.2f}, n={dvz_nav:.2f}, i={dvz_i:.2f}')
    return cmd


if __name__ == '__main__':
    # run tests
    pass

