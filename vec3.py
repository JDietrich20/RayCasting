import math

E = 1e-6 #Episilon for floating point comparisons

# Vector addition
def add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

# Vector subtraction
def subtract(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

# Scalar multiplication
def mul(v, s):
    return (v[0]*s, v[1]*s, v[2]*s)

# Dot product
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

# Cross product
def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

# Vector length
def length(v):
    return math.sqrt(dot(v, v))

# Normalize vector
def normalize(v):
    L = length(v)
    if L == 0:
        return (0.0, 0.0, 0.0)
    return (v[0]/L, v[1]/L, v[2]/L)