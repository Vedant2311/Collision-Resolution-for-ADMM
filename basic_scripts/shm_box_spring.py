################################################################################
# Basic Function to simulate SHM block system
# Uses CVXOPT solver for the system
# The spring's base length is zero here, with fixed point at origin
################################################################################

import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

import cvxopt
from cvxopt import matrix
import math
from cvxopt import solvers
import numpy as np

class block:
    
    ### Specify the parameters for your simulation here. 
    def __init__(self, length = 2, spring_constant = 5, initial_position = 4, mass = 10, dt = 1./100):
        self.vertices = [[0,0,0],[0,length,0],[length,length,0],[length,0,0]]
        self.edges = [[0,1],[1,2],[2,3],[3,0]]
        self.spring_constant = spring_constant
        self.length = length
        self.mass = mass
        self.current_position = initial_position
        self.current_velocity = 0
        self.updated_position = 0
        self.updated_velocity = 0
        self.dt = dt
        self.time_elasped = 0
        self.displacement = abs(initial_position)
        
    def update_position(self):
        self.updated_position = self.current_position + self.current_velocity*self.dt
        
    def update_velocity(self):
        self.updated_velocity = (self.updated_position - self.current_position)/self.dt
    
    def update_state(self):
    
    	## Computes the predicted positions for the box
    	## Uses convex optimiser to simulate spring forces
        self.update_position()
        p = np.zeros((1))
        p[0] = (self.mass/(self.dt**2) + self.spring_constant)
        q = np.zeros((1))
        q[0] = - self.mass*self.updated_position/(self.dt**2) 
        new_position = np.array(solvers.qp(matrix(p), matrix(q))['x'])
        
        ## Updating the new velocity
        self.updated_position = new_position
        self.update_velocity()
        self.current_position = self.updated_position
        self.current_velocity = self.updated_velocity
        self.time_elasped += self.dt
        
   	## Draws the Box at the corresponding time frame using it's coordinates
    def draw(self):
        glBegin(GL_QUADS)
        print("Position: ",self.current_position)
        self.vertices[0][0] = self.current_position
        self.vertices[1][0] = self.current_position
        self.vertices[2][0] = self.current_position + self.length
        self.vertices[3][0] = self.current_position + self.length
        for edge in self.edges:
            for vertex in edge:
                glVertex3fv(self.vertices[vertex])
        positional_vertices = [[-self.displacement,0,0],[-self.displacement,self.length,0],[0,0,0],[0,self.length,0],[self.displacement + self.length,0,0],[self.displacement + self.length,self.length,0]]
        positional_edges = [[0,1],[2,3],[4,5]]
        glEnd()
        glBegin(GL_LINES)
        for edge in positional_edges:
            for vertex in edge:
                glVertex3fv(positional_vertices[vertex])
        glEnd()
        
block = block()
def main():
    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0,0,-50)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        block.update_state()
        block.draw()
        pg.display.flip()
        pg.time.wait(10)

if __name__ == "__main__":
    main()