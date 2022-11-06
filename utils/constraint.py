import math
import numpy as np

class spherical_constraint():
    
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center
        self.lambda0 = 0
        
    def constraint_function(self, x):
        
        value = np.zeros((len(x)//3))
        center = self.center
        radius = self.radius
        for i in range(0,len(x),3):
            x1,y1,z1 = x[i:i+3]
            val = math.sqrt((x1 - center[0])**2 + (y1 - center[1])**2 + (z1 - center[2])**2) - radius
            value[i//3] = val
        
        return value
    
    def gradient_function(self, x):
        
        
        center = self.center
        value = np.zeros((len(x)//3, len(x)))
        for i in range(0,len(x),3):
            x1,y1,z1 = x[i:i+3]
            value[i//3][i] = 2*(x1 - center[0])
            value[i//3][i + 1] = 2*(y1 - center[1])
            value[i//3][i + 2] = 2*(z1 - center[2])
        return value
    
    #supports only one point
    def nearest_surface_point(self, x):
        
        x1, y1, z1 = x
        center = self.center
        radius = self.radius
        distance_required = max(0, radius - math.sqrt((x1 - center[0])**2 + (y1 - center[1])**2 + (z1 - center[2])**2))
        normal = np.sum(self.gradient_function(x), 0)
        normal = (normal/np.linalg.norm(normal)) if np.linalg.norm(normal) != 0 else normal
        new_point = x + distance_required * normal
        return new_point
    
class planar_constraint():
    
    def __init__(self, boundary_x = None, boundary_y = None, boundary_z = None):
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
        self.boundary_z = boundary_z
        self.lambda0 = 0
        
    def constraint_function(self, x):
        
        boundary_x = self.boundary_x
        boundary_y = self.boundary_y
        boundary_z = self.boundary_z
        value = np.ones((len(x)))
        for i in range(0,len(x),3):
            x1,y1,z1 = x[i:i+3]
            
            if boundary_x is not None:
                value[i] = (x1 - boundary_x)
            if boundary_y is not None:
                value[i + 1] = (y1 - boundary_y)
            if boundary_z is not None:
                value[i + 2] = (z1 - boundary_z)
        return value
    
    def gradient_function(self, x):
        
        if self.constraint_function(x).all() > -1e-3:
            return np.zeros((len(x),len(x)))
        boundary_x = self.boundary_x
        boundary_y = self.boundary_y
        boundary_z = self.boundary_z
        gradient = np.zeros((len(x),len(x)))
        
        if boundary_x is not None:
            for i in range(0, len(x), 3):
                gradient[i][i] = 1
        if boundary_y is not None:
            for i in range(1, len(x), 3):
                gradient[i][i] = 1
        if boundary_z is not None:
            for i in range(2, len(x), 3):
                gradient[i][i] = 1
            
        return gradient
    
    def nearest_surface_point(self, x):
        
        boundary_x = self.boundary_x
        boundary_y = self.boundary_y
        boundary_z = self.boundary_z
        
        new_point = x
        
        if boundary_x is not None:
            new_point[0] = boundary_x
        if boundary_y is not None:
            new_point[1] = boundary_y
        if boundary_z is not None:
            new_point[2] = boundary_z
            
        return new_point