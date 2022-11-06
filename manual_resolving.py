import sys
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import pickle
from utils.config import *
from utils.mesh import *
from utils.constraint import *
from utils.optimizer import *

L = LENGTH_OF_MESH/(NUMBER_OF_VERTICES_PER_ROW-1)

total_vertices = NUMBER_OF_VERTICES_PER_ROW**2
initial_position = np.zeros((total_vertices*3))
initial_velocity = np.zeros((total_vertices*3))

for i in range(NUMBER_OF_VERTICES_PER_ROW):
    for j in range(NUMBER_OF_VERTICES_PER_ROW):
        initial_position[3*(i*NUMBER_OF_VERTICES_PER_ROW + j):3*(i*NUMBER_OF_VERTICES_PER_ROW + j) + 3] =  [i*L +1 ,2*RADIUS, j*L - LENGTH_OF_MESH/2]

constraint_1 = ({'type': 'ineq', 'fun': spherical_constraint(RADIUS, np.zeros(3)).constraint_function, 'constraint' : spherical_constraint(RADIUS, np.zeros(3))}) 
constraint_2 = ({'type': 'ineq', 'fun': planar_constraint(boundary_y = 0).constraint_function, 'constraint' : planar_constraint(boundary_y = 0)}) 
        
mesh = Mesh(manual_resolving_optimizer, initial_position, initial_velocity, L = L, M = TOTAL_MASS/(NUMBER_OF_VERTICES_PER_ROW**2),
            G = GRAVITATIONAL_ACC, K = SPRING_CONSTANT, K_2 = SPRING_CONSTANT_2, num_vertices_in_row=NUMBER_OF_VERTICES_PER_ROW, 
            dt = TIME_STEP, constraints = [constraint_1, constraint_2])

def main():
    pg.init()
    display = (800, 600)
    window = pg.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(90, (display[0]/display[1]), 0.1, 50*LENGTH_OF_MESH)
    
    glLight(GL_LIGHT0, GL_POSITION,  (1e10,1e10,0, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1,1.,1, 1))
    glLightfv(GL_LIGHT1, GL_AMBIENT, (0.1,0.1,0.1, 1))
    
    glEnable(GL_DEPTH_TEST) 
    sphere = gluNewQuadric()
    
    glTranslatef(-10,-10,-30)
    glRotate(-45, 0,5,0)
    #glRotate(90, 5,0,0)
    img_id = 0
    while (True):
        #print( mesh.time_elapsed)
        for event in pg.event.get():
            if event.type == pg.QUIT or mesh.time_elapsed > TOTAL_TIME:
                pg.quit()
                sys.exit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        
        
        glBegin(GL_QUADS)
        glColor4f(0.8,0.8,0.8,1) 
        glNormal3d(0,1,0)
        glVertex3fv([ 500,-LENGTH_OF_MESH/10, 500])
        glVertex3fv([ 500,-LENGTH_OF_MESH/10,-500])
        glVertex3fv([-500,-LENGTH_OF_MESH/10,-500])
        glVertex3fv([-500,-LENGTH_OF_MESH/10, 500])
        glEnd()
        
        glColor4f(0.7, 0.7, 0.2, 1) #Put color
        gluSphere(sphere, RADIUS - LENGTH_OF_MESH/10, 50, 50) 
        mesh.update_state()
        mesh.draw()
        
        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHT1)
        glDisable(GL_LIGHTING)
        
        #Below is to save images
        #pg.image.save(window, os.path.join(__location__,"images/collision_scipy/image" + str(img_id) + ".png"))
        #img_id+=1
        pg.display.flip()
        pg.time.wait(100)

if __name__ == "__main__":
    main()
