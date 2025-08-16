import numpy as np

class Plane:
    def __init__(self):
        """Базовый класс для плоских поверхностей."""
        pass

class Floor(Plane):
    def __init__(self, y=-10):
        """
        Инициализация горизонтального пола.
        y: координата по вертикали.
        """
        super().__init__()
        self.y = y

class InclinedPlane(Plane):
    def __init__(self, x0=100, y0=-10, z0=-100, size_x=200, size_z=200, angle_deg=30):
        """
        Инициализация наклонной плоскости.
        x0, y0, z0: координаты начала плоскости.
        size_x, size_z: размеры плоскости по осям X и Z.
        angle_deg: угол наклона в градусах.
        """
        super().__init__()
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.size_x, self.size_z = size_x, size_z
        self.angle = np.radians(angle_deg)
        self.tan_a = np.tan(self.angle)
