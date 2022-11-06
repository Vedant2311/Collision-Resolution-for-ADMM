################################################################################
#   1. Energy optimization using CVXOPT (one norm term is left out, 
#   which comes due to non zero free length)
#	2. Using a large value of spring constant will squeeze the system,
#	because of the inherent zero free length assumption
################################################################################

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

frame = []

class Triangular_Mesh:
    
    def __init__(self,
                 L=1,  # length of one spring in m
                 M=0.01,  # mass of one particle in Kg
                 G=9.8,  # acceleration due to gravity, in m/s^2
                 K=0.1, #spring constant
                 num_vertices_in_row = 3, #vertices in a row
                 dt = 1./30 #time step
                 ): 
                 
        ## Initialising the parameters for the system
        self.num_vertices_in_row = num_vertices_in_row
        self.time_elapsed = 0
        self.dt = dt
        self.G = G
        self.M = M
        self.K = K
        self.L = L
        
        total_vertices = num_vertices_in_row**2
        self.total_vertices = total_vertices
        all_vertices = np.zeros((total_vertices,3))
        alpha = np.zeros((3*total_vertices,3*total_vertices))
        beta = np.zeros((3*total_vertices,3*total_vertices))
        gamma = np.zeros((3*total_vertices,3*total_vertices))
        all_edges = []
        
        ## Initial positions for the vertices
        for i in range(num_vertices_in_row):
            for j in range(num_vertices_in_row):
                all_vertices[i*num_vertices_in_row + j] = [j,0,i]
       	
       	## Constructing the edges for the system      
        for i in range(total_vertices):
            for j in range(i,total_vertices):
                
                x1,_,y1 = all_vertices[i]
                x2,_,y2 = all_vertices[j]
                if(x1==x2 and (y2-y1)==1):
                    all_edges.append((i,j))
                    alpha[3*i][3*j] = K
                    alpha[3*i+1][3*j+1] = K
                    alpha[3*i+2][3*j+2] = K
                    
                    alpha[3*j][3*i] = K
                    alpha[3*j+1][3*i+1] = K
                    alpha[3*j+2][3*i+2] = K
                    continue
                if(y1==y2 and (x2-x1)==1):
                    all_edges.append((i,j))
                    alpha[3*i][3*j] = K
                    alpha[3*i+1][3*j+1] = K
                    alpha[3*i+2][3*j+2] = K
                    
                    alpha[3*j][3*i] = K
                    alpha[3*j+1][3*i+1] = K
                    alpha[3*j+2][3*i+2] = K
                    continue
                    
                #Commented diagonal edges for now
                if(y2==y1+1 and x2==x1+1):
                    all_edges.append((i,j))
                    alpha[3*i][3*j] = K
                    alpha[3*i+1][3*j+1] = K
                    alpha[3*i+2][3*j+2] = K
                    
                    alpha[3*j][3*i] = K
                    alpha[3*j+1][3*i+1] = K
                    alpha[3*j+2][3*i+2] = K
                    continue
                
        self.all_vertices = all_vertices
        self.all_edges = all_edges
        
        #Modeling the equality constraint, Ax = B, where 2 points are fixed                
        A = np.zeros((6,total_vertices*3))
        B = np.zeros((6))
        
        A[0][0] = 1
        A[1][1] = 1
        A[2][2] = 1
        A[3][3*(num_vertices_in_row-1)] = 1
        A[4][3*(num_vertices_in_row-1)+1] = 1
        A[5][3*(num_vertices_in_row-1)+2] = 1
        
        B[0] = 0
        B[1] = 0
        B[2] = 0
        B[3] = 0
        B[4] = 0
        B[5] = L*(num_vertices_in_row-1)
        
        self.A = matrix(A)
        self.B = matrix(B)
        
        ### Constructing the position and velocity vectors 
        current_position = np.zeros((total_vertices*3))
        current_velocity = np.zeros((total_vertices*3))
        
        for i in range(total_vertices):
            gamma[3*i][3*i] = 1
            gamma[3*i+1][3*i+1] = 1
            gamma[3*i+2][3*i+2] = 1
        
        for i in range(num_vertices_in_row):
            for j in range(num_vertices_in_row):
                current_position[3*(i*num_vertices_in_row + j):3*(i*num_vertices_in_row + j) + 3] = [i*L,0,j*L]
        
        for i in range(total_vertices):
            for j in range(total_vertices):
                beta[3*i][3*i] += alpha[3*i][3*j]
                beta[3*i+1][3*i+1] += alpha[3*i+1][3*j+1]
                beta[3*i+2][3*i+2] += alpha[3*i+2][3*j+2]
                
      	## Creating mass matrix for gravity
        mass_matrix = np.zeros((3*total_vertices))  
        for i in range(total_vertices):
            mass_matrix[3*i+1] = 1
        mass_matrix*=M*G
        
        
        ## Fix related to the alpha matrix
        alpha = -alpha
        
        self.mass_matrix = mass_matrix
        gamma*= M/(dt*dt)     
        self.P = matrix(alpha + beta + gamma)
        
        self.current_position = current_position
        self.current_velocity = current_velocity
        self.updated_position = current_position
        self.updated_velocity = current_velocity
        
    def update_position(self):
        self.updated_position = self.current_position + self.current_velocity*self.dt 
        
    def update_velocity(self, new_position):
        dt = self.dt
        self.updated_velocity = (new_position - self.current_position)/dt
        
    def update_state(self):
        M = self.M
        dt = self.dt
        self.update_position()
        
        q = matrix(-M*self.updated_position/(dt*dt) + self.mass_matrix)
        new_position =  np.array(solvers.qp(self.P,q,A=self.A,b=self.B)['x']).reshape((3*self.total_vertices))
        self.update_velocity(new_position)
        self.current_position = new_position
        self.current_velocity = self.updated_velocity
        self.time_elapsed += dt
        frame.append(self.current_position)
        
    def draw(self):
        glBegin(GL_LINES)
        for edge in self.all_edges:
            for vertex in edge:
                x,y,z = self.current_position[3*vertex:3*vertex+3]
                glVertex3fv([x,y,z])
        glEnd()

def animate(mesh):
    mesh.update_state()

    
##################################################################
##                      CONFIG                                  ##
##################################################################

TOTAL_MASS = 0.05
SPRING_CONSTANT = 1
GRAVITATIONAL_ACC = 9.8
LENGTH_OF_MESH = 0.3
NUMBER_OF_VERTICES_PER_ROW = 3    
TIME_STEP = 0.003
TOTAL_TIME = 10
##################################################################


mesh = Triangular_Mesh(L = LENGTH_OF_MESH/(NUMBER_OF_VERTICES_PER_ROW), M = TOTAL_MASS/(NUMBER_OF_VERTICES_PER_ROW**2),G = GRAVITATIONAL_ACC, K = SPRING_CONSTANT, num_vertices_in_row=NUMBER_OF_VERTICES_PER_ROW, dt = TIME_STEP)

def main():
    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(15, (display[0]/display[1]), 0.1, 10*LENGTH_OF_MESH)

    glTranslatef(-LENGTH_OF_MESH/10,0,-5*LENGTH_OF_MESH)
    glRotate(-45, 0,5,0)
    while (True):
        
        for event in pg.event.get():
            if event.type == pg.QUIT or mesh.time_elapsed > TOTAL_TIME:
                print(len(frame))
                pg.quit()
                sys.exit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        animate(mesh)
        mesh.draw()
        pg.display.flip()
        pg.time.wait(1)

if __name__ == "__main__":
    main()
