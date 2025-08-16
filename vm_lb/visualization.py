import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from quaternions import quat_to_rotmat
from cube import Cube

class Visualization:
    def __init__(self, world):
        """Инициализация визуализации с объектом мира."""
        self.world = world
        self.traj = []               # хранение траектории
        self.linear_acc = np.zeros(3)
        self.angular_acc = np.zeros(3)

    def draw_axes(self, s=2):
        """Рисует координатные оси X (красная), Y (зелёная), Z (синяя)."""
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(s,0,0)
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,s,0)
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,s)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_floor(self):
        """Рисует горизонтальный пол на уровне y."""
        y = self.world.floor.y
        glDisable(GL_LIGHTING)
        glColor3f(0.3,0.6,0.3)
        glBegin(GL_QUADS)
        glVertex3f(-100,y,-100); glVertex3f(-100,y,100)
        glVertex3f(100,y,100); glVertex3f(100,y,-100)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_inclined_plane(self):
        """Рисует наклонную плоскость."""
        plane = self.world.inclined
        glDisable(GL_LIGHTING)
        glColor3f(1.0,0.5,0.0)
        x0, y0, z0 = plane.x0, plane.y0, plane.z0
        size_x, size_z, tan_a = plane.size_x, plane.size_z, plane.tan_a
        x1,y1,z1 = x0+size_x, y0+size_x*tan_a, z0
        x2,y2,z2 = x0+size_x, y0+size_x*tan_a, z0+size_z
        x3,y3,z3 = x0, y0, z0+size_z
        glBegin(GL_QUADS)
        glVertex3f(x0,y0,z0); glVertex3f(x1,y1,z1)
        glVertex3f(x2,y2,z2); glVertex3f(x3,y3,z3)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_cube(self):
        """Рисует куб с раскрашенными гранями."""
        hs = 0.7
        glBegin(GL_QUADS)
        # красная
        glColor3f(1.0, 0.0, 0.0); glNormal3f(1,0,0)
        glVertex3f(hs,-hs,-hs); glVertex3f(hs,hs,-hs)
        glVertex3f(hs,hs,hs); glVertex3f(hs,-hs,hs)
        # зелёная
        glColor3f(0.0, 1.0, 0.0); glNormal3f(-1,0,0)
        glVertex3f(-hs,-hs,-hs); glVertex3f(-hs,-hs,hs)
        glVertex3f(-hs,hs,hs); glVertex3f(-hs,hs,-hs)
        # синяя
        glColor3f(0.0,0.0,1.0); glNormal3f(0,1,0)
        glVertex3f(-hs,hs,-hs); glVertex3f(-hs,hs,hs)
        glVertex3f(hs,hs,hs); glVertex3f(hs,hs,-hs)
        # жёлтая
        glColor3f(1.0,1.0,0.0); glNormal3f(0,-1,0)
        glVertex3f(-hs,-hs,-hs); glVertex3f(hs,-hs,-hs)
        glVertex3f(hs,-hs,hs); glVertex3f(-hs,-hs,hs)
        # магента
        glColor3f(1.0,0.0,1.0); glNormal3f(0,0,1)
        glVertex3f(-hs,-hs,hs); glVertex3f(hs,-hs,hs)
        glVertex3f(hs,hs,hs); glVertex3f(-hs,hs,hs)
        # циан
        glColor3f(0.0,1.0,1.0); glNormal3f(0,0,-1)
        glVertex3f(-hs,-hs,-hs); glVertex3f(-hs,hs,-hs)
        glVertex3f(hs,hs,-hs); glVertex3f(hs,-hs,-hs)
        glEnd()

    def draw_striped_sphere(self, radius=0.7, slices=32, stacks=16):
        """Рисует сферу с чередующимися полосами (чёрная/белая)."""
        for i in range(stacks):
            lat0 = np.pi * (-0.5 + i / stacks)
            lat1 = np.pi * (-0.5 + (i + 1) / stacks)
            y0 = radius * np.sin(lat0)
            y1 = radius * np.sin(lat1)
            r0 = radius * np.cos(lat0)
            r1 = radius * np.cos(lat1)
            glColor3f(1,1,1) if i%2==0 else glColor3f(0,0,0)
            glBegin(GL_QUAD_STRIP)
            for j in range(slices+1):
                lng = 2*np.pi*j/slices
                x = np.cos(lng); z = np.sin(lng)
                glNormal3f(x*r0/radius, y0/radius, z*r0/radius)
                glVertex3f(x*r0, y0, z*r0)
                glNormal3f(x*r1/radius, y1/radius, z*r1/radius)
                glVertex3f(x*r1, y1, z*r1)
            glEnd()

    def draw_grid(self, step=50, size=100):
        """Рисует сетку на полу и наклонной плоскости."""
        glDisable(GL_LIGHTING)
        glColor3f(0,0,2)
        y = self.world.floor.y
        for i in range(-size,size+1,step):
            glBegin(GL_LINES)
            glVertex3f(i,y,-size); glVertex3f(i,y,size)
            glVertex3f(-size,y,i); glVertex3f(size,y,i)
            glEnd()
        plane = self.world.inclined
        x0, y0, z0 = plane.x0, plane.y0, plane.z0
        size_x, size_z, tan_a = plane.size_x, plane.size_z, plane.tan_a
        for i in range(0, size_x+1, step):
            glBegin(GL_LINES)
            glVertex3f(x0+i, y0+i*tan_a, z0); glVertex3f(x0+i, y0+i*tan_a, z0+size_z)
            glEnd()
        for i in range(0, size_z+1, step):
            glBegin(GL_LINES)
            glVertex3f(x0, y0, z0+i); glVertex3f(x0+size_x, y0+size_x*tan_a, z0+i)
            glEnd()
        glEnable(GL_LIGHTING)

    def draw_trajectory(self, traj):
        """Рисует траекторию движения объекта."""
        glDisable(GL_LIGHTING)
        glColor3f(1,1,0)
        glBegin(GL_LINE_STRIP)
        for p in traj: glVertex3f(*p)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_force_vector(self, body):
        """Рисует вектор силы, приложенной к объекту."""
        f = body.user_force
        if np.linalg.norm(f)<1e-6: return
        p0 = body.x
        p1 = p0 + f*0.05
        glDisable(GL_LIGHTING)
        glColor3f(1,0,0)
        glBegin(GL_LINES)
        glVertex3f(*p0); glVertex3f(*p1)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_hud(self, body, linear_acc, angular_acc):
        """Отображает информацию на экране (HUD)."""
        glDisable(GL_LIGHTING)
        glColor3f(1,1,1)
        glWindowPos2f(10,10)
        s1 = f"t={self.world.t:.2f}  x={body.x.round(2)}  v={body.v.round(2)}"
        s2 = f"a={linear_acc.round(2)}"
        s3 = f"w={body.w_body.round(2)}  alpha={angular_acc.round(2)}"
        for ch in s1+' '+s2+' '+s3:
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(ch))
        glEnable(GL_LIGHTING)

    def draw_body(self, body):
        """Рисует тело (куб или сферу) с учётом позиции и ориентации."""
        R = quat_to_rotmat(body.q)
        M = np.eye(4)
        M[:3,:3] = R
        M[:3,3] = body.x
        glPushMatrix()
        glMultMatrixf(M.T)
        if isinstance(body, Cube):
            self.draw_cube()
        else:
            self.draw_striped_sphere(body.radius)
        glPopMatrix()
