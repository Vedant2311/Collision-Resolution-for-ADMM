########################################################################################
#   Energy optimization using manual collision resolution, spherical obstacle at origin
########################################################################################
import multiprocessing
from multiprocessing import Process
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
import numpy as np
import os
from scipy import optimize
import pickle
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
frame = []

class optimizer():

    def __init__(self,A, b, x0, x_prev, vertices, edges, edges_2, L, M, G, K, K_2, dt, radius):
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
        self.radius = radius
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
            
        #print(energy)
        return energy   
    
    def con1(self, x):
        dist = []
        min_val = 10000000000000000000000
        for i in range(0,len(x),3):
            x1,y1,z1 = x[i:i+3]
            min_val = min(min_val, (math.sqrt((x1)**2 + y1**2 + (z1)**2) - self.radius))
        #return 10
        return min_val
    
    def con2(self,x):
        
        dist = []
        min_val = 10000000000000000000000
        for i in range(1,len(x),3):
            min_val = min(min_val, (x[i]))
        
        
        return min_val
    
    def resolve(self, x, i):
        x1,y1,z1 = x[i:i+3]
        x0 = self.x0
        if(self.con1([x1,y1,z1]) < 0):
            x1 = np.array(x0[i:i+3])
            x2 = np.array(x[i:i+3])
            
            collision_point = np.array(x2)
            
            start = np.array(x1)
            while(abs(self.con1(start))>1e-10):
                line_lambda = 0.5
                mid = line_lambda*start + (1-line_lambda)*collision_point
                if(self.con1(mid) > 0):
                    start = mid
                else:
                    collision_point = mid
                    
            collision_point = start
            
            h = 1e-10
            
            surface_normal_x = self.con1(x2) - self.con1(x2 + np.array([h,0,0]))
            surface_normal_y = self.con1(x2) - self.con1(x2 + np.array([0,h,0]))
            surface_normal_z = self.con1(x2) - self.con1(x2 + np.array([0,0,h]))
            surface_normal = -np.array([surface_normal_x,surface_normal_y,surface_normal_z])/h
            surface_normal /=(np.linalg.norm(surface_normal) + 1e-100)
            
            start = 0
            end = 1000
            while(abs(self.con1(x2 + (start + end)*surface_normal/2)) > 1e-10):
                if self.con1(x2 + (start + end)*surface_normal/2) > 0:
                    end = (start + end)/2
                else:
                    start = (start + end)/2
                    
            final_x = x2 + (start + end)*surface_normal
            
            x[i:i+3] = final_x
            #x0[i:i+3] = collision_point
            
        x1,y1,z1 = x[i:i+3]   
        if(self.con2([x1,y1,z1]) < 0):
            x1 = np.array(x0[i:i+3])
            x2 = np.array(x[i:i+3])
            
            collision_point = np.array(x2)
            start = np.array(x1)
            while(abs(self.con2(start))>1e-10):
                line_lambda = 0.5
                mid = line_lambda*start + (1-line_lambda)*collision_point
                if(self.con2(mid) > 0):
                    start = mid
                else:
                    collision_point = mid
            collision_point = start
              
            h = 1e-10
            surface_normal_x = self.con2(x2) - self.con2(x2 + np.array([h,0,0]))
            surface_normal_y = self.con2(x2) - self.con2(x2 + np.array([0,h,0]))
            surface_normal_z = self.con2(x2) - self.con2(x2 + np.array([0,0,h]))
            surface_normal = -np.array([surface_normal_x,surface_normal_y,surface_normal_z])/h
            surface_normal /=(np.linalg.norm(surface_normal) + 1e-100)
            
            start = 0
            end = 1000
            while(abs(self.con2(x2 + (start + end)*surface_normal/2)) > 1e-10):
                if self.con2(x2 + (start + end)*surface_normal/2) > 0:
                    end = (start + end)/2
                else:
                    start = (start + end)/2
                    
            final_x = x2 + (start + end)*surface_normal
            
            x[i:i+3] = final_x
            #x0[i:i+3] = collision_point
        return x

    def solver(self):
        
        optimum = optimize.minimize(self.objective_function, tol = 10**(-10), 
                                    x0 = self.x_prev)
        
        x = optimum.x
        x0 = self.x0
        if(self.con1(x) >= 0 and self.con2(x) >= 0):
            return optimum.x, self.x0
        else:
            
            for j in range(0,len(x), 30):
                np = multiprocessing.cpu_count()
                p_list=[]
                for i in range(j,j + 30, 3):
                    if i >= len(x):
                        break
                    #x = self.resolve(x, i)
                    
                    #print('You have {0:1d} CPUs'.format(np))
                
                    p = Process(target=self.resolve, name='Process'+str((i-j)//3), args=(x,i,))
                    p_list.append(p)
                    print('Process:: ', p.name,)
                    p.start()
                    print('Was assigned PID:: ', p.pid)
                
                # Wait for all the processes to finish
                for p in p_list:
                    p.join()
                    
            return x, x0
                
            
            

class Triangular_Mesh:
    
    def __init__(self,
                 L=1,  # length of edge in m
                 M=0.05,  # mass of each particle in kg
                 G=9.8,  # acceleration due to gravity, in m/s^2
                 K=10, #spring constant
                 K_2=1, #spring constant
                 num_vertices_in_row = 3, #
                 dt = 1./5 , #time step
                 radius = 1
                 ): 
        
        self.num_vertices_in_row = num_vertices_in_row
        self.time_elapsed = 0
        self.dt = dt
        self.G = G
        self.M = M
        self.K = K
        self.K_2 = K_2
        self.L = L
        self.radius = radius
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
                current_position[3*(i*num_vertices_in_row + j):3*(i*num_vertices_in_row + j) + 3] =  [i*L + 1,2*self.radius,L*(num_vertices_in_row-1)/2-j*L]
        
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
        opt = optimizer(self.A,self.B,self.current_position, self.updated_position,self.all_vertices, self.all_edges, self.all_edges_2, self.L, self.M, self.G, self.K,self.K_2, self.dt, self.radius)
        new_position, self.current_position = opt.result
        
        self.update_velocity(new_position)
        self.current_position = new_position
        self.current_velocity = self.updated_velocity
        
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
            v1 = np.array(triangle_v[2]) - np.array(triangle_v[0])
            v2 = np.array(triangle_v[0]) - np.array(triangle_v[1])
            vn = np.cross(v2,v1)
            vn/=(np.linalg.norm(vn) + 1e-100)
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
LENGTH_OF_MESH = 6
NUMBER_OF_VERTICES_PER_ROW = 3    
TIME_STEP = 0.1
TOTAL_TIME = 5
RADIUS = 10
##################################################################


mesh = Triangular_Mesh(L = LENGTH_OF_MESH/(NUMBER_OF_VERTICES_PER_ROW-1), M = TOTAL_MASS/(NUMBER_OF_VERTICES_PER_ROW**2),
                       G = GRAVITATIONAL_ACC, K = SPRING_CONSTANT, K_2 = SPRING_CONSTANT_2, num_vertices_in_row=NUMBER_OF_VERTICES_PER_ROW, dt = TIME_STEP, radius = RADIUS + min(LENGTH_OF_MESH,RADIUS)/10)


#BELOW IS TO SAVE THE FRAMES
while(mesh.time_elapsed <= TOTAL_TIME):
    animate(mesh)
    frame.append(mesh.current_position)
    print(mesh.time_elapsed)

