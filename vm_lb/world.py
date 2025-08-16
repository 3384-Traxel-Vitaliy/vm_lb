from cube import Cube
from sphere import Sphere
from plane import Floor, InclinedPlane
import numpy as np

class World:
    def __init__(self, shape="cube", g=np.array([0, -9.81, 0])):
        """Инициализация мира с телом, полом и наклонной плоскостью"""
        if shape == "cube":
            self.body = Cube()
        else:
            self.body = Sphere()

        self.floor = Floor()
        self.inclined = InclinedPlane()
        self.dt = 1/60
        self.t = 0
        self.traj = []
        self.g = g

    def step(self):
        """Выполнение одного шага симуляции"""
        self.body.user_force = self.body.mass * self.g
        self.body.rk4_step(self.t, self.dt)
        self.body.handle_floor_collision(self.floor)
        self.body.handle_inclined_collision(self.inclined)

        self.traj.append(self.body.x.copy())
        if len(self.traj) > 500:
            self.traj.pop(0)

        self.t += self.dt
