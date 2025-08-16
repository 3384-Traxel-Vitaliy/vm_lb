import numpy as np

def quat_normalize(q):
    """
    Нормализует кватернион.
    q: массив [w, x, y, z]
    Возвращает нормализованный кватернион.
    """
    return q / np.linalg.norm(q)

def quat_mul(q1, q2):
    """
    Умножение двух кватернионов.
    q1, q2: массивы [w, x, y, z]
    Возвращает произведение кватернионов.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_from_omega(omega):
    """
    Преобразует угловую скорость в кватернион.
    omega: вектор угловой скорости [wx, wy, wz]
    Возвращает кватернион [0, wx, wy, wz].
    """
    return np.array([0.0, *omega], dtype=float)

def quat_to_rotmat(q):
    """
    Преобразует кватернион в матрицу вращения 3x3.
    q: массив [w, x, y, z]
    Возвращает матрицу вращения.
    """
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),     2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z),   2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),     1-2*(x*x+y*y)]
    ], dtype=float)
