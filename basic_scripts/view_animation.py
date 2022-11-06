######################################################################
#   VIEW ANIMATIONS FROM SAVED FRAMES
######################################################################

import pygame as pg
from pygame.locals import *
import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from numpy import linalg as LA
import cvxopt
from cvxopt import matrix
import math
from cvxopt import solvers
import os
import numpy as np
from scipy import optimize
import pickle
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#VIEW = 'TWO CLIP'
#VIEW = 'ONE CLIP'
VIEW = 'SHADOW'
#VIEW = 'OBSTACLE'
#VIEW = 'OBSTACLE2'

if VIEW == 'TWO CLIP':
    TOTAL_MASS = 0.05
    SPRING_CONSTANT = 1
    GRAVITATIONAL_ACC = 9.8
    LENGTH_OF_MESH = 0.3
    NUMBER_OF_VERTICES_PER_ROW = 3    
    TIME_STEP = 0.05
    TOTAL_TIME = 5
    
    frame_file = open(os.path.join(__location__, 'data/two_clip_mesh_scipy/frame'), 'rb')
    frame = pickle.load(frame_file)                      
    frame_file.close() 
    
    mesh_file = open(os.path.join(__location__, 'data/two_clip_mesh_scipy/mesh'), 'rb') 
    mesh = pickle.load(mesh_file)                      
    mesh_file.close()
    
    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(90, (display[0]/display[1]), 0.1, 10*LENGTH_OF_MESH)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(-LENGTH_OF_MESH/10,0,-2*LENGTH_OF_MESH)
    glRotate(-45, 0,5,0)
    
    glLight(GL_LIGHT0, GL_POSITION,  (0, 0, 0, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 10, 1))
    
    glEnable(GL_DEPTH_TEST) 
    
    for i in range(len(frame)):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                print(len(frame))
                pg.quit()
                sys.exit()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT)
    
        mesh.current_position = frame[i]
        mesh.draw()
    
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)
        pg.display.flip()
        pg.time.wait(10)
    pg.quit()
    sys.exit()
    
elif VIEW=='ONE CLIP':
    
    TOTAL_MASS = 0.05
    SPRING_CONSTANT = 1
    GRAVITATIONAL_ACC = 9.8
    LENGTH_OF_MESH = 1
    NUMBER_OF_VERTICES_PER_ROW = 3    
    TIME_STEP = 0.05
    TOTAL_TIME = 20
    
    frame_file = open(os.path.join(__location__, 'data/one_clip_mesh_scipy/frame'), 'rb')
    frame = pickle.load(frame_file)                      
    frame_file.close() 
    
    mesh_file = open(os.path.join(__location__, 'data/one_clip_mesh_scipy/mesh'), 'rb') 
    mesh = pickle.load(mesh_file)                      
    mesh_file.close()
    
    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(90, (display[0]/display[1]), 0.1, 10*LENGTH_OF_MESH)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(-LENGTH_OF_MESH/10,0,-2*LENGTH_OF_MESH)
    glRotate(-45, 0,5,0)
    glLight(GL_LIGHT0, GL_POSITION,  (0, 0, 0, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (10, 0, 0, 1))
    
    glEnable(GL_DEPTH_TEST)
    
    for i in range(len(frame)):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                print(len(frame))
                pg.quit()
                sys.exit()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT)
    
        mesh.current_position = frame[i]
        mesh.draw()
    
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)
        
        pg.display.flip()
        pg.time.wait(1)
    pg.quit()
    sys.exit()
    
elif VIEW=='OBSTACLE':
    TOTAL_MASS = 0.05
    SPRING_CONSTANT = 1
    GRAVITATIONAL_ACC = 9.8
    LENGTH_OF_MESH = 10
    NUMBER_OF_VERTICES_PER_ROW = 5    
    TIME_STEP = 0.05
    TOTAL_TIME = 3
    RADIUS = 5
    frame_file = open(os.path.join(__location__, 'data/obstacle/frame'), 'rb')
    frame = pickle.load(frame_file)                      
    frame_file.close() 
    
    mesh_file = open(os.path.join(__location__, 'data/obstacle/mesh'), 'rb') 
    mesh = pickle.load(mesh_file)                      
    mesh_file.close()
    
    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(90, (display[0]/display[1]), 0.1, 10*LENGTH_OF_MESH)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0,-5,-2*LENGTH_OF_MESH)
    #glRotate(-45, 0,5,0)
    #glRotate(45, 5,5,0)
    
    glLight(GL_LIGHT0, GL_POSITION,  (0, 0, 0, 1)) # point light from the left, top, front
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 10, 1))
    #glLightfv(GL_LIGHT0, GL_DIFFUSE, (10, 0, 0, 10))
    
    glEnable(GL_DEPTH_TEST) 
    sphere = gluNewQuadric() #Create new sphere
    for i in range(0,7000,5):
        for event in pg.event.get():
            if event.type == pg.QUIT or i > 6990:
                print(len(frame))
                print(i)
                pg.quit()
                sys.exit()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glTranslatef(0,0,0)
        glColor4f(0.5, 0.2, 0.2, 1) #Put color
        gluSphere(sphere, RADIUS, 48, 48) 
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT)
        
        mesh.current_position = frame[i]
        mesh.draw()
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)
        
        glColor3f(1,1,1)
        glBegin(GL_QUADS)
        glColor3f(1,1,1)
        glVertex3fv([100,0,100])
        glColor3f(1,1,1)
        glVertex3fv([100,0,-100])
        glColor3f(1,1,1)
        glVertex3fv([-100,0,-100])
        glColor3f(1,1,1)
        glVertex3fv([-100,0,100])
        glEnd()
        
        pg.display.flip()
        pg.time.wait(1)
    pg.quit()
    sys.exit()
    
    
elif VIEW=='OBSTACLE2':
    
    TOTAL_MASS = 0.05
    SPRING_CONSTANT = 1
    GRAVITATIONAL_ACC = 9.8
    LENGTH_OF_MESH = 10
    NUMBER_OF_VERTICES_PER_ROW = 5    
    TIME_STEP = 0.05
    TOTAL_TIME = 3
    RADIUS = 3
    frame_file = open(os.path.join(__location__, 'data/obstacle/frame_2'), 'rb')
    frame = pickle.load(frame_file)                      
    frame_file.close() 
    
    mesh_file = open(os.path.join(__location__, 'data/obstacle/mesh_2'), 'rb') 
    mesh = pickle.load(mesh_file)                      
    mesh_file.close()
    
    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(90, (display[0]/display[1]), 0.1, 10*LENGTH_OF_MESH)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0,-5,-2*LENGTH_OF_MESH)
    #glRotate(-45, 0,5,0)
    #glRotate(45, 5,5,0)
    
    glLight(GL_LIGHT0, GL_POSITION,  (0, 0, 0, 1)) # point light from the left, top, front
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 10, 1))
    #glLightfv(GL_LIGHT0, GL_DIFFUSE, (10, 0, 0, 10))
    
    glEnable(GL_DEPTH_TEST) 
    sphere = gluNewQuadric() #Create new sphere
    for i in range(0,len(frame),5):
        for event in pg.event.get():
            if event.type == pg.QUIT or i > 6990:
                print(len(frame))
                print(i)
                pg.quit()
                sys.exit()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glTranslatef(0,0,0)
        glColor4f(0.5, 0.2, 0.2, 1) #Put color
        gluSphere(sphere, RADIUS, 48, 48) 
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT)
        
        mesh.current_position = frame[i]
        mesh.draw()
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)
        
        glColor3f(1,1,1)
        glBegin(GL_QUADS)
        glColor3f(1,1,1)
        glVertex3fv([100,0,100])
        glColor3f(1,1,1)
        glVertex3fv([100,0,-100])
        glColor3f(1,1,1)
        glVertex3fv([-100,0,-100])
        glColor3f(1,1,1)
        glVertex3fv([-100,0,100])
        glEnd()
        
        pg.display.flip()
        pg.time.wait(10)
    pg.quit()
    sys.exit()
    
elif VIEW=='SHADOW':
    
    TOTAL_MASS = 0.05
    SPRING_CONSTANT = 1
    GRAVITATIONAL_ACC = 9.8
    LENGTH_OF_MESH = 10
    NUMBER_OF_VERTICES_PER_ROW = 8    
    TIME_STEP = 0.05
    TOTAL_TIME = 3
    RADIUS = 3
    frame_file = open(os.path.join(__location__, 'data/obstacle/frame_lighting'), 'rb')
    frame = pickle.load(frame_file)                      
    frame_file.close() 
    
    mesh_file = open(os.path.join(__location__, 'data/obstacle/mesh_lighting'), 'rb') 
    mesh = pickle.load(mesh_file)                      
    mesh_file.close()
    
    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(90, (display[0]/display[1]), 0.1, 50*LENGTH_OF_MESH)
    #glMatrixMode(GL_MODELVIEW)
    
    
    glLight(GL_LIGHT0, GL_POSITION,  (1000,0,0, 1)) # point light from the left, top, front
    
    
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1,1,1,1))
    
    #glLight1(GL_LIGHT1, GL_POSITION,  (100, 1000, 0, 1)) # point light from the left, top, front
    
    glLightfv(GL_LIGHT1, GL_AMBIENT, (0.2,0.2,0.2, 1))
    glEnable(GL_DEPTH_TEST) 
    sphere = gluNewQuadric() #Create new sphere
    
    glTranslatef(0,-10,-20)
    for i in range(0,len(frame),3):
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                print(len(frame))
                pg.quit()
                sys.exit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glBegin(GL_QUADS)
        glColor4f(0.5,0.5,0.5,1) 
        glVertex3fv([50,0,50])
        glVertex3fv([50,0,-50])
        glVertex3fv([-50,0,-50])
        glVertex3fv([-50,0,50])
        glNormal3d(0,1,0)
        glEnd()
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        
          
        
        glColor4f(1, 0., 0., 1) #Put color
        gluSphere(sphere, RADIUS, 48, 48) 
        
    
        mesh.current_position = frame[i]
        mesh.draw()
        
           
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHT1)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)
        pg.display.flip()
        pg.time.wait(1)
    pg.quit()
    sys.exit()


        
    
    