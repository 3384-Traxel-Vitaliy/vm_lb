import numpy as np
from body import Body
from plane import Floor, InclinedPlane
from quaternions import *


class Cube(Body):
    EPSILON = 1e-8

    def __init__(self, I_diag=(0.3, 0.4, 0.5), planes=None, half_size=0.7):
        """Инициализация куба с моментами инерции, плоскостями для столкновений и половиной размера."""
        super().__init__(mass=1.0, I_diag=I_diag)
        self.half_size = float(half_size)
        self.planes = planes if planes else []

    @staticmethod
    def _project_on_plane(vector, plane_normal):
        """Проекция вектора на плоскость с заданной нормалью."""
        return vector - np.dot(vector, plane_normal) * plane_normal

    @staticmethod
    def _safe_norm(vector):
        """Вычисление нормы вектора."""
        return np.linalg.norm(vector)

    def handle_floor_collision(self, floor, restitution=0.5, friction=0.3, dt=1/60.0):
        """Обработка столкновения с горизонтальным полом."""
        self._handle_floor_collision(floor, restitution, friction, dt)

    def handle_inclined_collision(self, plane, restitution=0.5, friction=0.3, dt=1/60.0):
        """Обработка столкновения с наклонной плоскостью."""
        self._handle_inclined_collision(plane, restitution, friction, dt)

    def handle_collisions(self, restitution=0.5, friction=0.3, dt=1/60.0):
        """Обработка всех столкновений с плоскостями, заданными в self.planes."""
        for plane in self.planes:
            if isinstance(plane, Floor):
                self._handle_floor_collision(plane, restitution, friction, dt)
            elif isinstance(plane, InclinedPlane):
                self._handle_inclined_collision(plane, restitution, friction, dt)

    def _handle_floor_collision(self, floor, restitution, friction, dt):
        """Внутренняя логика обработки столкновения с полом."""
        if not self._is_within_floor_bounds():
            return
        penetration = self._compute_floor_penetration(floor)
        if penetration <= 0:
            return
        self._resolve_penetration(penetration, np.array([0.0, 1.0, 0.0]), restitution, friction, dt)

    def _handle_inclined_collision(self, plane, restitution, friction, dt):
        """Внутренняя логика обработки столкновения с наклонной плоскостью."""
        if not self._is_within_plane_bounds(plane):
            return
        plane_normal = self._compute_plane_normal(plane)
        penetration = self._compute_inclined_penetration(plane, plane_normal)
        if penetration <= 0:
            return
        self._resolve_penetration(penetration, plane_normal, restitution, friction, dt)

    def _compute_floor_penetration(self, floor):
        """Вычисление глубины проникновения куба в пол."""
        return floor.y + self.half_size - self.x[1]

    def _compute_inclined_penetration(self, plane, plane_normal):
        """Вычисление глубины проникновения куба в наклонную плоскость."""
        plane_y = plane.y0 + (self.x[0] - plane.x0) * plane.tan_a
        return plane_y + self.half_size - self.x[1]

    def _resolve_penetration(self, penetration, plane_normal, restitution, friction, dt):
        """Разрешение проникновения: смещение, коррекция скорости и вращения."""
        self.x += plane_normal * penetration
        self._resolve_collision(plane_normal, restitution, friction)
        self._check_and_apply_rolling(plane_normal, friction, dt)

    def _resolve_collision(self, plane_normal, restitution, friction):
        """Коррекция линейной и угловой скорости при столкновении."""
        normal_velocity = np.dot(self.v, plane_normal) * plane_normal
        tangent_velocity = self.v - normal_velocity
        if np.dot(normal_velocity, plane_normal) < 0:
            self.v -= (1.0 + restitution) * normal_velocity
        self.v -= friction * tangent_velocity
        self.w_body *= (1.0 - friction)
        self._align_to_plane(plane_normal)

    def _align_to_plane(self, plane_normal):
        """Выравнивание ориентации куба вдоль нормали плоскости."""
        right_vec, forward_vec = self._compute_plane_axes(plane_normal)
        rotation_matrix = np.column_stack((right_vec, plane_normal, forward_vec))
        self.q = self._rotation_matrix_to_quat(rotation_matrix)

    @staticmethod
    def _rotation_matrix_to_quat(R):
        """Преобразование матрицы вращения в кватернион."""
        m00, m01, m02 = R[0]
        m10, m11, m12 = R[1]
        m20, m21, m22 = R[2]
        trace = m00 + m11 + m22
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (m21 - m12) / S
            y = (m02 - m20) / S
            z = (m10 - m01) / S
        elif (m00 > m11) and (m00 > m22):
            S = np.sqrt(1.0 + m00 - m11 - m22) * 2
            w = (m21 - m12) / S
            x = 0.25 * S
            y = (m01 + m10) / S
            z = (m02 + m20) / S
        elif m11 > m22:
            S = np.sqrt(1.0 + m11 - m00 - m22) * 2
            w = (m02 - m20) / S
            x = (m01 + m10) / S
            y = 0.25 * S
            z = (m12 + m21) / S
        else:
            S = np.sqrt(1.0 + m22 - m00 - m11) * 2
            w = (m10 - m01) / S
            x = (m02 + m20) / S
            y = (m12 + m21) / S
            z = 0.25 * S
        return np.array([w, x, y, z])

    def _compute_plane_axes(self, plane_normal):
        """Вычисление локальных осей плоскости для ориентации куба."""
        right_vec = np.array([1.0, 0.0, 0.0])
        forward_vec = np.cross(plane_normal, right_vec)
        if self._safe_norm(forward_vec) < self.EPSILON:
            forward_vec = np.array([0.0, 0.0, 1.0])
        right_vec = np.cross(forward_vec, plane_normal)
        return right_vec, forward_vec

    def _is_within_floor_bounds(self):
        """Проверка, находится ли куб в пределах пола."""
        x, _, z = self.x
        return -100.0 <= x <= 100.0 and -100.0 <= z <= 100.0

    def _is_within_plane_bounds(self, plane):
        """Проверка, находится ли куб в пределах наклонной плоскости."""
        x, _, z = self.x
        return plane.x0 <= x <= plane.x0 + plane.size_x and plane.z0 <= z <= plane.z0 + plane.size_z

    @staticmethod
    def _compute_plane_normal(plane):
        """Вычисление нормали наклонной плоскости."""
        normal = np.array([-plane.tan_a, 1.0, 0.0], dtype=float)
        return normal / np.linalg.norm(normal)

    def _check_and_apply_rolling(self, plane_normal, friction_coeff, dt):
        """Проверка условий для катания и применение крутящего момента."""
        gravity = np.array([0.0, -9.81, 0.0])
        tangential_force_vec = self._project_on_plane(gravity, plane_normal)
        tangential_force = self.mass * self._safe_norm(tangential_force_vec)
        normal_force = self.mass * np.dot(-gravity, plane_normal)
        if friction_coeff * normal_force >= tangential_force and self._is_center_above_edge(plane_normal):
            self._apply_edge_rolling(plane_normal, dt, rolling_friction=friction_coeff)

    def _is_center_above_edge(self, plane_normal):
        """Проверка, находится ли центр куба выше края плоскости для катания."""
        edge_height = self.x[1] - self.half_size
        return self.x[1] > edge_height + 1e-3

    def _apply_edge_rolling(self, plane_normal, dt=1/60.0, rolling_friction=0.02):
        """Применение катящегося движения куба по краю плоскости."""
        rolling_radius = self.half_size * np.sqrt(2)/2
        v_tangent = self._project_on_plane(self.v, plane_normal)
        speed = self._safe_norm(v_tangent)
        if speed < self.EPSILON:
            return
        direction = v_tangent / speed
        rotation_axis = np.cross(plane_normal, direction)
        rotation_axis /= self._safe_norm(rotation_axis)
        angular_speed = speed / rolling_radius
        friction_torque = -rotation_axis * angular_speed * rolling_friction
        angular_speed += np.dot(friction_torque, rotation_axis) * dt
        angular_speed = max(0.0, angular_speed)
        self.w_body = rotation_axis * angular_speed
        self.v = direction * angular_speed * rolling_radius
