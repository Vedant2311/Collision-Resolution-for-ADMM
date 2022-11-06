import math
from scipy import optimize
import numpy as np

def objective_function(optimizer, x):
    energy = 0
    edges = optimizer.mesh.all_edges
    edges_2 = optimizer.mesh.all_edges_2
    vertices = optimizer.mesh.all_vertices
    
    K = optimizer.mesh.K
    K_2 = optimizer.mesh.K_2
    L = optimizer.mesh.L
    M = optimizer.mesh.M
    G= optimizer.mesh.G
    dt = optimizer.mesh.dt
    
    x_updated = optimizer.mesh.updated_position
    #spring potential energy
    for i in range(len(edges)):
        v1 = x[3*edges[i][0]:3*edges[i][0] + 3]
        v2 = x[3*edges[i][1]:3*edges[i][1] + 3]
        
        ver1 = vertices[edges[i][0]]
        ver2 = vertices[edges[i][1]]
        
        distance = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)
        if(ver1[0]==ver2[0] or ver1[2] == ver2[2]):
            energy += 0.5*K*(distance - L )**2
        else:
            energy += 0.5*K*(distance - L*math.sqrt(2))**2
            
    for i in range(len(edges_2)):
        v1 = x[3*edges_2[i][0]:3*edges_2[i][0] + 3]
        v2 = x[3*edges_2[i][1]:3*edges_2[i][1] + 3]
        
        ver1 = vertices[edges_2[i][0]]
        ver2 = vertices[edges_2[i][1]]
        
        distance = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)
        #print(distance, 2*self.L)
        if(ver1[0]==ver2[0] or ver1[2] == ver2[2]):
            energy += 0.5*K_2*(distance - 2*L)**2
        else:
            energy += 0.5*K_2*(distance - 2*L*math.sqrt(2))**2
        
    #gravitational potential
            
    for i in range(1,len(x),3):
        energy += M*G*x[i]
       
    #kinetic component
        
    for i in range(len(x)):
        energy += (0.5*M*(x[i] - x_updated[i])**2)/(dt**2)
        
    return energy

def objective_gradient(optimizer, x):
    gradient = np.zeros((len(x)))
    edges = optimizer.mesh.all_edges
    edges_2 = optimizer.mesh.all_edges_2
    vertices = optimizer.mesh.all_vertices
    
    K = optimizer.mesh.K
    K_2 = optimizer.mesh.K_2
    L = optimizer.mesh.L
    M = optimizer.mesh.M
    G= optimizer.mesh.G
    dt = optimizer.mesh.dt
    
    x_updated = optimizer.mesh.updated_position
    #spring potential gradient
    for i in range(len(edges)):
        v1 = x[3*edges[i][0]:3*edges[i][0] + 3]
        v2 = x[3*edges[i][1]:3*edges[i][1] + 3]
        
        ver1 = vertices[edges[i][0]]
        ver2 = vertices[edges[i][1]]
        
        distance = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)
        if(ver1[0]==ver2[0] or ver1[2] == ver2[2]):
            gradient[3*edges[i][0]:3*edges[i][0] + 3] += K*(distance - L)*(v1 - v2)/distance
            gradient[3*edges[i][1]:3*edges[i][1] + 3] += K*(distance - L)*(v2 - v1)/distance
        else:
            gradient[3*edges[i][0]:3*edges[i][0] + 3] += K*(distance - L*math.sqrt(2))*(v1 - v2)/distance
            gradient[3*edges[i][1]:3*edges[i][1] + 3] += K*(distance - L*math.sqrt(2))*(v2 - v1)/distance
            
    for i in range(len(edges_2)):
        v1 = x[3*edges_2[i][0]:3*edges_2[i][0] + 3]
        v2 = x[3*edges_2[i][1]:3*edges_2[i][1] + 3]
        
        ver1 = vertices[edges_2[i][0]]
        ver2 = vertices[edges_2[i][1]]
        
        distance = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)
        if(ver1[0]==ver2[0] or ver1[2] == ver2[2]):
            gradient[3*edges_2[i][0]:3*edges_2[i][0] + 3] += K_2*(distance - 2*L)*(v1 - v2)/distance
            gradient[3*edges_2[i][1]:3*edges_2[i][1] + 3] += K_2*(distance - 2*L)*(v2 - v1)/distance
        else:
            gradient[3*edges_2[i][0]:3*edges_2[i][0] + 3] += K_2*(distance - 2*L*math.sqrt(2))*(v1 - v2)/distance
            gradient[3*edges_2[i][1]:3*edges_2[i][1] + 3] += K_2*(distance - 2*L*math.sqrt(2))*(v2 - v1)/distance
        
    #gravitational potential
            
    for i in range(1,len(x),3):
        gradient[i] += M*G
       
    #kinetic component
        
    for i in range(len(x)):
        gradient[i] += (M*(x[i] - x_updated[i]))/(dt**2)
        
    return gradient

def objective_function_1(optimizer, x, constraints, x0):
    
    energy = objective_function(optimizer, x)
    for constraint_dict in constraints:
        constraint = constraint_dict['constraint']
        
        if np.min(constraint.constraint_function(x0)) > 0:
            continue
        a = np.dot((np.sum(constraint.gradient_function(x0), 0)),(x - x0))*constraint.lambda0
        # print(a.shape)
        # print((np.sum(constraint.gradient_function(x0), 0)).reshape(1,-1).shape)
        
        # print((x - x0).reshape(-1,1).shape)
        energy += a
    return energy

def objective_gradient_1(optimizer, x, constraints, x0):
    gradient = objective_gradient(optimizer, x)
    for constraint_dict in constraints:
        
        constraint = constraint_dict['constraint']
        if np.min(constraint.constraint_function(x0)) > 0:
            continue
        gradient += np.transpose(np.sum(constraint.gradient_function(x0), 0))*constraint.lambda0
    
    return gradient

class scipy_optimizer():

    def __init__(self, mesh):
        self.mesh = mesh
        self.result = self.solver()
        
    def objective(self, x):
        return objective_function(self, x)
        
    def objective_gradient(self, x):
        return objective_gradient(self, x)

    def solver(self):
        constraints = self.mesh.constraints
        optimum = optimize.minimize(self.objective, jac = self.objective_gradient, tol = 10**(-3),  x0 = self.mesh.current_position, constraints = constraints)
        #print(optimize.check_grad(self.objective, self.objective_gradient, self.mesh.current_position)) #to check gradient
        x = optimum.x
        return x
    
class manual1_resolving_optimizer():

    def __init__(self, mesh):
        self.mesh = mesh
        self.x0 = mesh.current_position
        self.constraints = mesh.constraints
        self.result = self.solver()
        
    def objective(self, x):
        return objective_function_1(self, x, self.constraints, self.x0)
        
    def objective_gradient(self, x):
        return objective_gradient_1(self, x, self.constraints, self.x0)

    def solver(self):
        constraints = self.mesh.constraints
        optimum = optimize.minimize(self.objective, jac = self.objective_gradient, tol = 10**(-3), x0 = self.mesh.current_position)
        x = optimum.x
        dt = self.mesh.dt ** 2
        M = self.mesh.M
        violated_constraints = []
        for constraint_dict in constraints:
            constraint = constraint_dict['constraint']
            function = constraint.constraint_function
            if np.min(function(x)) < 0:
                violated_constraints.append(constraint)
        
        for i in range(0,len(x),3):
            
            point = x[i:i+3]
            violated_constraints = []
            for constraint_dict in constraints:
                constraint = constraint_dict['constraint']
                function = constraint.constraint_function
                if np.min(function(point)) < -1e-3:
                    violated_constraints.append(constraint)
            
            for constraint in violated_constraints:
                lambda0 = constraint.lambda0
                new_point = constraint.nearest_surface_point(point)
                x[i:i+3] = new_point
                displacement = new_point - point
                normal = np.linalg.norm(constraint.gradient_function(new_point))
                del_lambda = M*np.linalg.norm(displacement)/(normal * dt) if normal !=0 else 0
                lambda0 += del_lambda
                constraint.lamba0 = lambda0    
        return x, constraints
    
class manual_resolving_optimizer():

    def __init__(self, mesh):
        self.mesh = mesh
        self.result = self.solver()
        
    def objective(self, x):
        return objective_function(self, x)
        
    def objective_gradient(self, x):
        return objective_gradient(self, x)

    def solver(self):
        constraints = self.mesh.constraints
        optimum = optimize.minimize(self.objective, jac = self.objective_gradient, tol = 10**(-3), x0 = self.mesh.current_position)
        x = optimum.x
        
        violated_constraints = []
        for constraint_dict in constraints:
            constraint = constraint_dict['constraint']
            function = constraint.constraint_function
            if np.min(function(x)) < 0:
                violated_constraints.append(constraint)
        
        for i in range(0,len(x),3):
            
            point = x[i:i+3]
            violated_constraints = []
            for constraint_dict in constraints:
                constraint = constraint_dict['constraint']
                function = constraint.constraint_function
                if np.min(function(point)) < -1e-3:
                    violated_constraints.append(constraint)
            
            for constraint in violated_constraints:
                new_point = constraint.nearest_surface_point(point)
                x[i:i+3] = new_point
                point = x[i:i+3]
        
        return x