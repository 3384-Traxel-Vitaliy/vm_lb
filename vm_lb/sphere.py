import numpy as np
from body import Body


class Sphere(Body):
    DEFAULT_GRAVITY = np.array([0.0, -9.81, 0.0])
    EPSILON = 1e-8
    PURE_ROLLING_FACTOR = 5.0 / 7.0

    def __init__(self, radius=2.0, mass=1.0, restitution=0.4, rolling_friction=0.01, gravity=None):
        """Инициализация сферы с заданными физическими параметрами."""
        super().__init__(mass, I_diag=((2.0 / 5.0) * mass * radius ** 2,) * 3)
        self.radius = float(radius)
        self.restitution = float(restitution)
        self.rolling_friction = float(rolling_friction)
        self.gravity = np.array(gravity) if gravity is not None else self.DEFAULT_GRAVITY
        self._epsilon = self.EPSILON

    @staticmethod
    def _project_vector_on_plane(vector, plane_normal):
        """Возвращает проекцию вектора на плоскость с нормалью plane_normal."""
        return vector - np.dot(vector, plane_normal) * plane_normal

    @staticmethod
    def _vector_norm(vector):
        """Возвращает норму (длину) вектора."""
        return np.linalg.norm(vector)

    def get_contact_velocity(self, contact_offset=None):
        """Возвращает скорость точки контакта сферы с учётом вращения."""
        if contact_offset is None:
            contact_offset = np.array([0.0, -self.radius, 0.0])
        return self.v + np.cross(self.w_body, contact_offset)

    def apply_gravity(self, dt: float):
        """Применяет гравитацию к сфере за интервал времени dt."""
        self.v += self.gravity * dt

    def _compute_angular_velocity_candidate(self, plane_normal, tangent_velocity):
        """Вычисляет кандидата на угловую скорость для чистого качения по плоскости."""
        omega = np.cross(plane_normal, tangent_velocity) / self.radius
        right_vector = np.cross(tangent_velocity, plane_normal)
        right_norm = self._vector_norm(right_vector)
        if right_norm > self._epsilon and np.dot(omega, right_vector / right_norm) > 0:
            omega = -omega
        return omega

    def _set_pure_rolling(self, plane_normal, tangent_velocity):
        """Устанавливает угловую скорость сферы для чистого качения по плоскости."""
        if self._vector_norm(tangent_velocity) < self._epsilon:
            if self._vector_norm(self.w_body) < 1e-6:
                self.w_body = np.zeros_like(self.w_body)
            return
        self.w_body = self._compute_angular_velocity_candidate(plane_normal, tangent_velocity)

    def _apply_rolling_resistance(self, plane_normal, dt: float):
        """Применяет сопротивление качению к сфере на плоскости."""
        omega_norm = self._vector_norm(self.w_body)
        if omega_norm < 1e-12:
            return
        normal_force = abs(self.mass * np.dot(-self.gravity, plane_normal))
        rolling_moment_magnitude = self.rolling_friction * normal_force * self.radius
        rolling_moment = -rolling_moment_magnitude * (self.w_body / omega_norm)
        try:
            angular_acceleration = self.I_body_inv @ rolling_moment
        except Exception:
            I_scalar = (2.0 / 5.0) * self.mass * (self.radius ** 2)
            angular_acceleration = rolling_moment / I_scalar
        self.w_body += angular_acceleration * dt

    def _resolve_normal_collision(self, contact_velocity, plane_normal):
        """Обрабатывает коллизию по нормали с коэффициентом упругости."""
        normal_velocity = np.dot(contact_velocity, plane_normal) * plane_normal
        if np.dot(normal_velocity, plane_normal) < 0:
            self.v -= (1.0 + self.restitution) * normal_velocity

    def _compute_tangent_velocity(self, plane_normal):
        """Возвращает компоненту скорости сферы, касательную к плоскости."""
        return self._project_vector_on_plane(self.v, plane_normal)

    def handle_floor_collision(self, floor, dt=1 / 60.0):
        """Обрабатывает столкновение сферы с горизонтальным полом."""
        up_normal = np.array([0.0, 1.0, 0.0])
        penetration_depth = floor.y + self.radius - self.x[1]
        if penetration_depth <= 0:
            return
        self.x[1] += penetration_depth
        self._process_floor_contact(up_normal, dt)

    def _process_floor_contact(self, up_normal, dt: float):
        """Вспомогательная функция для обработки контакта с полом."""
        contact_point = -self.radius * up_normal
        contact_velocity = self.get_contact_velocity(contact_point)
        self._resolve_normal_collision(contact_velocity, up_normal)
        tangent_velocity_center = self._compute_tangent_velocity(up_normal)
        self._set_pure_rolling(up_normal, tangent_velocity_center)
        self._apply_rolling_resistance(up_normal, dt)

    def handle_inclined_collision(self, plane, dt=1 / 60.0):
        """Обрабатывает столкновение сферы с наклонной плоскостью."""
        if not self._is_within_plane_bounds(plane):
            return
        plane_normal = self._get_plane_normal(plane)
        tangent_up = self._compute_tangent_up_vector(plane_normal)
        if self._vector_norm(tangent_up) < self._epsilon:
            return
        penetration_depth = self._compute_plane_penetration(plane, plane_normal)
        if penetration_depth <= 0:
            return
        self.x += plane_normal * penetration_depth
        self._process_inclined_contact(plane_normal, tangent_up, dt)

    def _process_inclined_contact(self, plane_normal, tangent_up, dt: float):
        """Вспомогательная функция для обработки контакта с наклонной плоскостью."""
        tangent_velocity = self._compute_tangent_velocity(plane_normal)
        contact_point = -self.radius * plane_normal
        contact_velocity = self.get_contact_velocity(contact_point)
        self._resolve_normal_collision(contact_velocity, plane_normal)
        tangent_projection = np.dot(tangent_velocity, tangent_up)
        if tangent_projection > 1e-6:
            self._roll_up_inclined(plane_normal, dt)
        elif tangent_projection < -1e-6:
            self._roll_down_inclined(plane_normal, dt)
        else:
            self._set_pure_rolling(plane_normal, tangent_velocity)
            self._apply_rolling_resistance(plane_normal, dt)

    def _is_within_plane_bounds(self, plane):
        """Проверяет, находится ли сфера в пределах плоскости по X и Z."""
        x, _, z = self.x
        return plane.x0 <= x <= plane.x0 + plane.size_x and plane.z0 <= z <= plane.z0 + plane.size_z

    def _get_plane_normal(self, plane):
        """Возвращает нормаль наклонной плоскости."""
        normal = np.array([-plane.tan_a, 1.0, 0.0], dtype=float)
        return normal / self._vector_norm(normal)

    def _compute_tangent_up_vector(self, plane_normal):
        """Вычисляет касательную единичную векторную ось вверх вдоль плоскости."""
        up_vector = np.array([0.0, 1.0, 0.0])
        tangent_up = self._project_vector_on_plane(up_vector, plane_normal)
        norm = self._vector_norm(tangent_up)
        if norm > self._epsilon:
            tangent_up /= norm
        return tangent_up

    def _compute_plane_penetration(self, plane, plane_normal):
        """Вычисляет глубину проникновения сферы в плоскость."""
        x, y, _ = self.x
        plane_y = plane.y0 + (x - plane.x0) * plane.tan_a
        return plane_y + self.radius - y

    def _roll_down_inclined(self, plane_normal, dt: float):
        """Реализует движение вниз по наклонной плоскости с вращением."""
        tangent_acceleration = self.PURE_ROLLING_FACTOR * self._project_vector_on_plane(self.gravity, plane_normal)
        self.v += tangent_acceleration * dt
        self._update_rolling(plane_normal)

    def _roll_up_inclined(self, plane_normal, dt: float):
        """Реализует движение вверх по наклонной плоскости с вращением."""
        tangent_acceleration = -self.PURE_ROLLING_FACTOR * self._project_vector_on_plane(self.gravity, plane_normal)
        self.v += tangent_acceleration * dt
        self._update_rolling(plane_normal)

    def _update_rolling(self, plane_normal):
        """Обновляет угловую скорость и сопротивление качению после движения вдоль плоскости."""
        tangent_velocity = self._compute_tangent_velocity(plane_normal)
        self._set_pure_rolling(plane_normal, tangent_velocity)
        self._apply_rolling_resistance(plane_normal, dt=1 / 60.0)

    def debug_state(self):
        """Возвращает текущее состояние сферы для отладки."""
        return {"x": self.x.copy(),
                "v": self.v.copy(),
                "w": self.w_body.copy()}
