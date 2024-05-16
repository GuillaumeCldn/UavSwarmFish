import numpy as np
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


@dataclass
class State:
    pos: np.ndarray
    speed: np.ndarray
    heading: float
    timestamp: float

    #TODO getter functions for velocity, distance, heading diff, etc

@dataclass
class Commands:
    delta_heading: float
    delta_speed: float
    delta_vz: float


def interaction_wall(agent: State, dist: float, angle: float):

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

