import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

from utils.config import *
class Mesh:
    
    def __init__(self, 
                 optimizer, 
                 initial_position, 
                 initial_velocity, 
                 L=1,  # length of edge in m
                 M=0.05,  # mass of each particle in kg
                 G=9.8,  # acceleration due to gravity, in m/s^2
                 K=10, #spring constant
                 K_2=1, #spring constant for weaker edges
                 num_vertices_in_row = 3, #particles in row
                 dt = 1./5 , #time step
                 constraints = None,
                 manual_method = 0
                 ): 
        
        self.num_vertices_in_row = num_vertices_in_row
        self.time_elapsed = 0
        self.dt = dt
        self.G = G
        self.M = M
        self.K = K
        self.K_2 = K_2
        self.L = L
        self.optimizer = optimizer
        self.constraints = constraints
        self.manual_method = manual_method
        total_vertices = num_vertices_in_row**2
        
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
            for j in range(i+1,total_vertices):
                
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
        #----------Edges, Vertices, Triangles Updated
        
        self.current_position = initial_position
        self.current_velocity = initial_velocity
        
        self.updated_position = initial_position
        
    def update_position(self):
        self.updated_position = self.current_position + self.current_velocity*self.dt 
        
    def update_velocity(self, new_position):
        self.current_velocity = (new_position - self.current_position)/self.dt
        
    def update_state(self):
        self.update_position()
        if self.manual_method == 1:
            new_position, self.constraints = self.optimizer(self).result
        else:
            new_position = self.optimizer(self).result
        #only for scenario 2
        # idx = 3*(NUMBER_OF_VERTICES_PER_ROW**2 - 1)//2
        # new_position[idx] = 0
        # new_position[idx+2] = 0
        self.update_velocity(new_position)
        self.current_position = new_position
        self.time_elapsed += self.dt
        
    def draw(self):
        
        cnt = 0
        for triangle in self.all_triangles:
            triangle_v = []
            glBegin(GL_TRIANGLES)
            glColor4f(0.2, 0.2, 0.8, 1) #Put color
            cnt += 1
            for vertex in triangle:
                x,y,z = self.current_position[3*vertex:3*vertex+3]
                
                triangle_v.append([x,y,z])
            v1 = np.array(triangle_v[2]) - np.array(triangle_v[0])
            v2 = np.array(triangle_v[1]) - np.array(triangle_v[0])
            vn = np.cross(v2,v1)
            vn/= (np.linalg.norm(vn))
            #print(cnt, " ", vn)
            glNormal3d(vn[0],vn[1],vn[2])
            for vertex in triangle_v:
                glVertex3fv(vertex)
            glEnd()
    