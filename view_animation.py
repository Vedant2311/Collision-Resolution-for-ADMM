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



frame_file = open(('rendered_info/frame_scipy_1'), 'rb')
frame = pickle.load(frame_file)                      
frame_file.close() 

mesh_file = open(('rendered_info/mesh_scipy_1'), 'rb')
mesh = pickle.load(mesh_file)                      
mesh_file.close()
"""

frame_file = open(('rendered_info/frame_manual_1'), 'rb')
frame = pickle.load(frame_file)                      
frame_file.close() 

mesh_file = open(('rendered_info/mesh_manual_1'), 'rb')
mesh = pickle.load(mesh_file)                      
mesh_file.close()
"""
def main():
    pg.init()
    display = (800, 600)
    window = pg.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(90, (display[0]/display[1]), 0.1, 50*LENGTH_OF_MESH)
    
    glLight(GL_LIGHT0, GL_POSITION,  (1e10,1e10,0, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.7,0.7,0.7, 1))
    glLightfv(GL_LIGHT1, GL_AMBIENT, (0.1,0.1,0.1, 1))
    
    glEnable(GL_DEPTH_TEST) 
    sphere = gluNewQuadric()
    
    glTranslatef(-10,-10,-30)
    glRotate(-45, 0,5,0)
    #glRotate(90, 5,0,0)
    img_id = 0
    for i in range(len(frame)):

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        
        
        #scenario 1 ======================================
        glBegin(GL_QUADS)
        glColor4f(0.8,0.8,0.8,1) 
        glNormal3d(0,1,0)
        glVertex3fv([ 500,-LENGTH_OF_MESH/10, 500])
        glVertex3fv([ 500,-LENGTH_OF_MESH/10,-500])
        glVertex3fv([-500,-LENGTH_OF_MESH/10,-500])
        glVertex3fv([-500,-LENGTH_OF_MESH/10, 500])
        glEnd()
        #=================================================
        glColor4f(0.7, 0.7, 0.2, 1) #Put color
        gluSphere(sphere, RADIUS - min(RADIUS,LENGTH_OF_MESH)/10, 50, 50)
        
        mesh.current_position = frame[i]
        mesh.draw()
        
        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHT1)
        glDisable(GL_LIGHTING)
        
        #Below is to save images
        pg.image.save(window, ("images/scipy_10x10/image" + str(img_id) + ".png"))
        img_id+=1
        pg.display.flip()
        pg.time.wait(20)
    pg.quit()
    sys.exit()

if __name__ == "__main__":
    main()
