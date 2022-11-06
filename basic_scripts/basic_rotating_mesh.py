#######################################################################
# Basic OpenGL program
# Rotates cloth around two fixed points
# Dependencies: Pygame and PyOpenGL (Install them using pip)
#######################################################################

import pygame
from pygame.locals import *
import sys
from OpenGL.GL import *
from OpenGL.GLU import *

num_vertices = 20
vertices = []
edges = []

# Creating the basic layout for the cloth
for i in range(num_vertices):
    for j in range(num_vertices):
        vertices.append((i,j,0))

for i in range(len(vertices)):
    for j in range(i,len(vertices)):
        x1,y1,_ = vertices[i]
        x2,y2,_ = vertices[j]
        if(x1==x2 and (y2-y1)==1):
            edges.append((i,j))
            continue
        if(y1==y2 and (x2-x1)==1):
            edges.append((i,j))
            continue
        if(y2==y1+1 and x2==x1+1):
            edges.append((i,j))
            continue
            
# Function to display the mesh
def Mesh():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(30, 1, 0.1, 500.0)
    glTranslatef(0.0,0.0,-100.0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        glRotatef(1, 1, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Mesh()
        pygame.display.flip()
        pygame.time.wait(10)

main()