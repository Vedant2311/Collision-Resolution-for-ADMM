######################################################################
#   Energy optimization using scipy
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
from numpy import linalg as LA
import math
import os
import numpy as np
from scipy import optimize
import pickle
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
frame = []

class optimizer():

    def __init__(self,A, b, x0, x_prev, vertices, edges, edges_2, L, M, G, K, K_2, dt):
        self.A    = A
        self.b    = b
        self.x0   = x0
        self.edges = edges
        self.edges_2 = edges_2
        # call solver
        self.G = G
        self.M = M
        self.K = K
        self.K_2 = K_2
        self.L = L
        self.dt = dt
        self.vertices = vertices
        self.x_prev=x_prev
        self.result = self.solver()

    def objective_function(self, x):
        energy = 0
        edges = self.edges
        edges_2 = self.edges_2
        K_2 = self.K_2
        #spring potential energy
        for i in range(len(self.edges)):
            v1 = x[3*edges[i][0]:3*edges[i][0] + 3]
            v2 = x[3*edges[i][1]:3*edges[i][1] + 3]
            
            ver1 = self.vertices[edges[i][0]]
            ver2 = self.vertices[edges[i][1]]
            
            distance = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)
            if(ver1[0]==ver2[0] or ver1[2] == ver2[2]):
                energy += 0.5*self.K*(distance - self.L )**2
            else:
                energy += 0.5*self.K*(distance - self.L*math.sqrt(2))**2
                
        for i in range(len(self.edges_2)):
            v1 = x[3*edges_2[i][0]:3*edges_2[i][0] + 3]
            v2 = x[3*edges_2[i][1]:3*edges_2[i][1] + 3]
            
            ver1 = self.vertices[edges_2[i][0]]
            ver2 = self.vertices[edges_2[i][1]]
            
            distance = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)
            #print(distance, 2*self.L)
            if(ver1[0]==ver2[0] or ver1[2] == ver2[2]):
                energy += 0.5*self.K_2*(distance - 2*self.L)**2
            else:
                energy += 0.5*self.K_2*(distance - 2*self.L*math.sqrt(2))**2
                
        
        #gravitational potential
                
        for i in range(1,len(x),3):
            energy += self.M*self.G*x[i]
           
        #kinetic component
            
        for i in range(len(x)):
            energy += (0.5*self.M*(x[i] - self.x_prev[i])**2)/(self.dt**2)
            
        return energy   
    
    def con(self, x):
        return np.dot(self.A, x).reshape(self.b.shape) - self.b

    def solver(self):
        cons = ({'type': 'eq', 'fun': self.con})
        optimum = optimize.minimize(self.objective_function, tol = 10**(-100), 
                                    x0 = self.x_prev, constraints = cons, callback=frame.append)
        return optimum.x

class Triangular_Mesh:
    
    def __init__(self,
                 L=1,  # length of edge in m
                 M=0.05,  # mass of each particle in kg
                 G=9.8,  # acceleration due to gravity, in m/s^2
                 K=10, #spring constant
                 K_2=1, #spring constant
                 num_vertices_in_row = 3, #
                 dt = 1./5 #time step
                 ): 
        
        self.num_vertices_in_row = num_vertices_in_row
        self.time_elapsed = 0
        self.dt = dt
        self.G = G
        self.M = M
        self.K = K
        self.K_2 = K_2
        self.L = L
        total_vertices = num_vertices_in_row**2
        self.total_vertices = total_vertices
        all_vertices = np.zeros((total_vertices,3))
        all_edges = []
        all_edges_2 = []
        all_triangles = []
        for i in range(num_vertices_in_row):
            for j in range(num_vertices_in_row):
                all_vertices[i*num_vertices_in_row + j] = [j,0,i]
                if i < num_vertices_in_row - 1 and j < num_vertices_in_row - 1:
                    all_triangles.append([i*num_vertices_in_row + j, i*num_vertices_in_row + j+1, (i+1)*num_vertices_in_row + j+1])
                    
                    all_triangles.append([(i+1)*num_vertices_in_row + j, i*num_vertices_in_row + j,(i+1)*num_vertices_in_row + j+1])
                
        
                
        for i in range(total_vertices):
            for j in range(i,total_vertices):
                
                x1,_,y1 = all_vertices[i]
                x2,_,y2 = all_vertices[j]
                if(x1==x2 and (y2-y1)==1):
                    all_edges.append((i,j))
                    continue
                if(y1==y2 and (x2-x1)==1):
                    all_edges.append((i,j))
                    continue
                if(y2==y1+1 and x2==x1+1):
                    all_edges.append((i,j))
                    continue
                
                
                #below is second layer of edges (weaker)
                if(x1==x2 and (y2-y1)==2):
                    all_edges_2.append((i,j))
                    continue
                if(y1==y2 and (x2-x1)==2):
                    all_edges_2.append((i,j))
                    continue
                if(y2==y1+2 and x2==x1+2):
                    all_edges_2.append((i,j))
                    continue
                
                
                
                
        self.all_vertices = all_vertices
        self.all_edges = all_edges
        self.all_edges_2 = all_edges_2
        self.all_triangles = all_triangles
        #----------Edges and Vertices Updated
        
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
        B[5] = 0 -L*(num_vertices_in_row-1)
        
        self.A = (A)
        self.B = (B)
        
        #----------A and B Updated
        
        current_position = np.zeros((total_vertices*3))
        current_velocity = np.zeros((total_vertices*3))
        
        
        for i in range(num_vertices_in_row):
            for j in range(num_vertices_in_row):
                current_position[3*(i*num_vertices_in_row + j):3*(i*num_vertices_in_row + j) + 3] = [i*L,0,0-j*L]
        
        self.current_position = current_position
        self.current_velocity = current_velocity
        self.updated_position = current_position
        self.updated_velocity = current_velocity
        
        self.initial_position = self.current_position
        
    def update_position(self):
        self.updated_position = self.current_position + self.current_velocity*self.dt 
        
    def update_velocity(self, new_position):
        dt = self.dt
        self.updated_velocity = (new_position - self.current_position)/dt
        
    def update_state(self):
        self.update_position()
        opt = optimizer(self.A,self.B,self.current_position, self.updated_position,self.all_vertices, self.all_edges, self.all_edges_2, self.L, self.M, self.G, self.K,self.K_2, self.dt)
        new_position = opt.result
        self.update_velocity(new_position)
        self.current_velocity = self.updated_velocity
        self.current_position = new_position
        self.time_elapsed += self.dt
        
    def draw(self):
        glBegin(GL_TRIANGLES)
        glColor4f(0, 0., 1., 1) #Put color
        for triangle in self.all_triangles:
            triangle_v = []
            for vertex in triangle:
                x,y,z = self.current_position[3*vertex:3*vertex+3]
                glVertex3fv([x,y,z])
                triangle_v.append([x,y,z])
            v1 = np.array(triangle_v[0]) - np.array(triangle_v[1])
            v2 = np.array(triangle_v[0]) - np.array(triangle_v[2])
            vn = np.cross(v2,v1)
            vn/=np.linalg.norm(vn)
            glNormal3d(vn[0],vn[1],vn[2])
        glEnd()

def animate(mesh):
    mesh.update_state()
    

##################################################################
##                      CONFIG                                  ##
##################################################################

TOTAL_MASS = 0.05
SPRING_CONSTANT = 1
SPRING_CONSTANT_2 = 1/10
GRAVITATIONAL_ACC = 9.8
LENGTH_OF_MESH = 1
NUMBER_OF_VERTICES_PER_ROW = 10    
TIME_STEP = 0.05
TOTAL_TIME = 10
##################################################################


mesh = Triangular_Mesh(L = LENGTH_OF_MESH/(NUMBER_OF_VERTICES_PER_ROW), M = TOTAL_MASS/(NUMBER_OF_VERTICES_PER_ROW**2),
                       G = GRAVITATIONAL_ACC, K = SPRING_CONSTANT, K_2 = SPRING_CONSTANT_2, num_vertices_in_row=NUMBER_OF_VERTICES_PER_ROW, dt = TIME_STEP)

"""
#BELOW IS TO SAVE THE FRAMES
while(mesh.time_elapsed <= TOTAL_TIME):
    animate(mesh)
    print(mesh.time_elapsed)
    
frame = np.array(frame).astype(float)
frame_file = open(os.path.join(__location__, 'data/two_clip_mesh_scipy/frame'), 'ab')
  
# source, destination 
pickle.dump(frame, frame_file)                      
frame_file.close() 

mesh_file = open(os.path.join(__location__, 'data/two_clip_mesh_scipy/mesh'), 'ab') 
pickle.dump(mesh, mesh_file)                      
mesh_file.close()

"""


def main():
    pg.init()
    display = (800, 600)
   
    window = pg.display.set_mode(display, DOUBLEBUF|OPENGL)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(90, (display[0]/display[1]), 0.1, 10*LENGTH_OF_MESH)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(-LENGTH_OF_MESH/10,0,-2*LENGTH_OF_MESH)
    glRotate(-70, 0,5,0)
    glLight(GL_LIGHT0, GL_POSITION,  (1000,1000,0, 1)) # point light from the left, top, front
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1,1,1,1))
    glLightfv(GL_LIGHT1, GL_AMBIENT, (0.2,0.2,0.2, 1))
    glEnable(GL_DEPTH_TEST) 
    
    img_id = 0
    
    if not os.path.exists(os.path.join(__location__,"images/two_clip/")):
        os.makedirs(os.path.join(__location__,"images/two_clip/"))
    while (True):
        
        for event in pg.event.get():
            if event.type == pg.QUIT or mesh.time_elapsed > TOTAL_TIME:
                print(len(frame))
                pg.quit()
                sys.exit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
        animate(mesh)
        mesh.draw()
    
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHT1)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)
        #pg.image.save(window, os.path.join(__location__,"images/two_clip/image" + str(img_id) + ".png"))
        img_id+=1
        pg.display.flip()
        pg.time.wait(5)

if __name__ == "__main__":
    main()
    