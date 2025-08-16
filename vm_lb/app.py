import sys
import time
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from world import World
from visualization import Visualization

class OpenGLApp:
    def __init__(self):
        """Инициализация OpenGL приложения и камеры"""
        self.world = World()
        self.vis = Visualization(self.world)
        self.cam_yaw = 0
        self.cam_pitch = 0
        self.cam_distance = 15
        self.last = time.time()
        self.traj = []
        self.state_history = []
        self.prev_v = np.zeros(3)
        self.prev_w = np.zeros(3)
        self.linear_acc = np.zeros(3)
        self.angular_acc = np.zeros(3)

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(1024,768)
        glutCreateWindow(b"Rigid Body Simulation")

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.07,0.07,0.1,1)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60,1024/768,0.1,1000)

        glutDisplayFunc(self.on_display)
        glutIdleFunc(self.on_idle)
        glutKeyboardFunc(self.on_key)
        glutSpecialFunc(self.on_special_key)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (5,5,10,1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.9,0.9,0.9,1))
        glEnable(GL_COLOR_MATERIAL)

    def on_key(self, key, x, y):
        """Обработка обычных клавиш управления телом и симуляцией"""
        body = self.world.body
        if key==b'\x1b': sys.exit(0)
        if key==b'w': body.user_force[0]+=10
        if key==b's': body.user_force[0]-=10
        if key==b'a': body.user_force[2]-=10
        if key==b'd': body.user_force[2]+=10
        if key==b'r': body.user_force[1]+=10
        if key==b'f': body.user_force[1]-=10
        if key==b'q': body.user_torque[1]+=1
        if key==b'e': body.user_torque[1]-=1
        if key ==b' ':
            self.step_forward()
        if key ==b'm':
            self.step_backward()
        if key in [b'x',b'X']:
            body.__init__()
            self.world.traj.clear()
            self.world.t=0
        if key in [b'z',b'Z']:
            body.user_force = np.zeros(3)
            body.user_torque = np.zeros(3)
        if key in [b'c',b'C']:
            try:
                coords = input("Введите новые координаты куба x y z через пробел: ")
                x,y,z = map(float, coords.split())
                body.x = np.array([x,y,z])
                body.v = np.zeros(3)
                body.w_body = np.zeros(3)
                body.user_force = np.zeros(3)
                body.user_torque = np.zeros(3)
                print(f"Куб перемещён в: {body.x}")
            except Exception as e:
                print("Неверный ввод. Используйте формат: x y z")

    def on_special_key(self, key, x, y):
        """Обработка специальных клавиш (стрелки) для управления камерой"""
        step_angle = np.radians(5)
        step_height = 1.0
        if key==GLUT_KEY_LEFT: self.cam_yaw -= step_angle
        if key==GLUT_KEY_RIGHT: self.cam_yaw += step_angle
        if key==GLUT_KEY_UP: self.cam_pitch = min(self.cam_pitch+step_height, 89)
        if key==GLUT_KEY_DOWN: self.cam_pitch = max(self.cam_pitch-step_height, -10)

    def step_forward(self):
        """Шаг симуляции вперёд с обновлением состояния тела"""
        body = self.world.body
        self.state_history.append((
            body.x.copy(), body.v.copy(), body.q.copy(), body.w_body.copy()
        ))
        body.rk4_step(self.world.t, self.world.dt)
        body.handle_floor_collision(self.world.floor)
        body.handle_inclined_collision(self.world.inclined)

        self.traj.append(body.x.copy())
        if len(self.traj) > 500:
            self.traj.pop(0)

        self.world.t += self.world.dt
        glutPostRedisplay()
        self.linear_acc = (body.v - self.prev_v) / self.world.dt
        self.prev_v = body.v.copy()
        self.angular_acc = (body.w_body - self.prev_w) / self.world.dt
        self.prev_w = body.w_body.copy()

    def step_backward(self):
        """Шаг симуляции назад с восстановлением предыдущего состояния"""
        if self.state_history:
            body = self.world.body
            body.x, body.v, body.q, body.w_body = self.state_history.pop()
            if self.traj:
                self.traj.pop()
            self.world.t -= self.world.dt
            glutPostRedisplay()

    def on_idle(self):
        """Функция idle, вызывается когда нет других событий"""
        glutPostRedisplay()

    def on_display(self):
        """Отрисовка сцены OpenGL"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        target = self.world.body.x
        x_offset = self.cam_distance * np.cos(np.radians(self.cam_pitch)) * np.sin(self.cam_yaw)
        y_offset = self.cam_distance * np.sin(np.radians(self.cam_pitch))
        z_offset = self.cam_distance * np.cos(np.radians(self.cam_pitch)) * np.cos(self.cam_yaw)
        eye = target + np.array([x_offset, y_offset, z_offset])
        gluLookAt(*eye, *target, 0, 1, 0)

        self.vis.draw_axes()
        self.vis.draw_floor()
        self.vis.draw_inclined_plane()
        self.vis.draw_grid()
        self.vis.draw_body(self.world.body)
        self.vis.draw_trajectory(self.traj)
        self.vis.draw_force_vector(self.world.body)
        self.vis.draw_hud(self.world.body, self.linear_acc, self.angular_acc)
        glutSwapBuffers()

def main():
    """Главная функция для запуска приложения"""
    choice = input("Выберите фигуру (cube/sphere): ").strip().lower()
    if choice not in ["cube", "sphere"]:
        choice = "cube"
    app = OpenGLApp()
    app.world = World(shape=choice)
    app.vis = Visualization(app.world)
    glutMainLoop()

if __name__ == "__main__":
    main()
