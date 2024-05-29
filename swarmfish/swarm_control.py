import numpy as np
import math
from dataclasses import dataclass

@dataclass
class SwarmParams:
    # collective motion parameters
    max_velocity: float
    min_velocity: float
    velocity: float
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
    yacc: float
    lacc: float
    dv0: float
    ew1_ob: float
    ew2_ob: float
    yob: float
    lob: float
    y_perp: float
    dz_perp: float
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
        fw = math.exp(-(dist / params.lw)**2)
        ow = params.ew1 * math.cos(angle) + params.ew2 * math.cos(2. * angle)
        cmd.delta_course = params.yw * math.sin(angle) * (1. + ow) * fw
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
    attenuation = 1. - math.exp(-(r_w/params.lw)**2)        # wall attenuation
    dij = agent.get_distance_coupled(other, params)         # distance between agents
    # alignment
    dphi = agent.get_course_diff(other, params.use_heading) # course/heading difference
    dyaw_ali = params.yali * ((dij + params.d0ali) / params.d0ali) * math.exp(-(dij/params.lali)**2) * math.sin(dphi) * attenuation
    # attraction
    psi = agent.get_viewing_angle(other, params.use_heading) # viewing angle
    dyaw_att = params.yatt * ((dij / params.d0att - 1.) / (1. + (dij / params.latt)**2)) * math.sin(psi) * attenuation
    #print(f'dyaw_att {np.degrees(dyaw_att):0.2f} | dyaw_ali {np.degrees(dyaw_ali):0.2f} | social {np.degrees(dyaw_att+dyaw_ali):.2f}')
    # speed
    #FIXME normalize dv0 - dij
    dv = params.yacc * math.cos(psi) * (params.dv0 - dij) / (1. + dij / params.lacc)
    # vertical
    #FIXME L_z_2 name
    dz = other.pos[2] - agent.pos[2]
    dvz = params.y_z * math.tanh((dz - math.copysign(params.dz0, dz)) / params.a_z) * math.exp(-(dij / params.L_z_2)**2)
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
    dyaw = -params.y_intruder * math.exp(-(dij / params.l_intruder)**2) * (1. + params.ew1 * math.cos(psi)) * math.sin(dphi)
    dz = agent.pos[2] - other.pos[2]
    dvz = params.y_z_intruder * math.tanh(dz / params.a_z) * math.exp(-(dij / params.L_z_2)**2) #FIXME name of L_z_2
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


# TODO make a better obstacle class

if __name__ == '__main__':
    # run tests
    pass

