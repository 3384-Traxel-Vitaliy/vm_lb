import numpy as np
from quaternions import *

class Body:
    def __init__(self, mass=1.0, I_diag=(0.3,0.4,0.5)):
        """Инициализация тела с массой, моментом инерции и начальными состояниями"""
        self.mass = mass
        self.I_body = np.diag(I_diag)
        self.I_body_inv = np.diag(1.0/np.array(I_diag))
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1,0,0,0], dtype=float)
        self.w_body = np.zeros(3)
        self.g = np.array([0, -9.81, 0])
        self.user_force = np.zeros(3)
        self.user_torque = np.zeros(3)

    def external_force_world(self, t):
        """Возвращает внешнюю силу в мировой системе координат"""
        return self.mass*self.g + self.user_force

    def external_torque_body(self, t):
        """Возвращает внешний момент сил в системе тела"""
        return self.user_torque

    def deriv(self, t, state):
        """Вычисляет производные состояния (скорости и ускорения)"""
        x, v, q, w = state
        a = self.external_force_world(t)/self.mass
        Iw = self.I_body @ w
        tau = self.external_torque_body(t)
        w_dot = self.I_body_inv @ (tau - np.cross(w, Iw))
        q_dot = 0.5 * quat_mul(q, quat_from_omega(w))
        return (v, a, q_dot, w_dot)

    def rk4_step(self, t, dt):
        """
        Выполняет один шаг интегрирования методом Рунге-Кутты 4-го порядка.

        t : float
            Текущее время.
        dt : float
            Шаг интегрирования.

        Метод обновляет:
            - self.x : положение тела
            - self.v : скорость тела
            - self.q : ориентацию (кватернион)
            - self.w_body : угловую скорость в теле
        """
        s0 = (self.x.copy(), self.v.copy(), self.q.copy(), self.w_body.copy())

        def add(s, ds, k):
            # Складывает состояния с масштабированными приращениями
            return tuple(si + k * di for si, di in zip(s, ds))

        k1 = self.deriv(t, s0)
        k2 = self.deriv(t + dt / 2, add(s0, k1, dt / 2))
        k3 = self.deriv(t + dt / 2, add(s0, k2, dt / 2))
        k4 = self.deriv(t + dt, add(s0, k3, dt))

        def combine(s, k1, k2, k3, k4):
            # Комбинирует приращения k1..k4 для расчета нового состояния
            return tuple(
                si + dt / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
                for si, d1, d2, d3, d4 in zip(s, k1, k2, k3, k4)
            )

        self.x, self.v, self.q, self.w_body = combine(s0, k1, k2, k3, k4)
        self.q = quat_normalize(self.q)  # нормализация кватерниона

