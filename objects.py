from vec3 import E, add, subtract, dot, mul, normalize, cross
import math

# Object defintions, intersection methods, and normal calculations for various shapes.

# Object definition of a Sphere
class Sphere:
    def __init__(self, location, radius, color, texture=None):
        self.location = location
        self.radius = radius
        self.color = color
        self.texture = texture

    def intersect(self, ray_origin, ray_direction):
        oc = subtract(ray_origin, self.location)
        a = dot(ray_direction, ray_direction)
        b = 2.0 * dot(oc, ray_direction)
        c = dot(oc, oc) - self.radius * self.radius
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        sqrt_d = math.sqrt(disc)
        t0 = (-b - sqrt_d) / (2*a)
        t1 = (-b + sqrt_d) / (2*a)
        t = None
        if t0 > E:
            t = t0
        elif t1 > E:
            t = t1
        return t

    def get_normal(self, point):
        n = subtract(point, self.location)
        return normalize(n)


# Object definition of a Plane
class Plane:
    def __init__(self, point=(0,0,0), normal=(0.0, 1.0, 0.0), color=(255,255,255), texture=None, uv_scale=1.0):
        self.point = point
        self.normal = normalize(normal) 
        self.color = color
        self.texture = texture
        self.uv_scale = uv_scale

    def intersect(self, ray_origin, ray_dir):
        denom = dot(ray_dir, self.normal)
        if abs(denom) < E:
            return None
        t = dot(subtract(self.point, ray_origin), self.normal) / denom
        if t < E:
            return None
        return t

    def get_normal(self, hit_point=None, extra=None):
        return self.normal


# Object definition of a Pyramid
class Pyramid:
    def __init__(self, base_center, base_size, height, color=(200,200,200)):
        half = base_size / 2.0
        x, y, z = base_center

        v0 = (x - half, y, z - half) 
        v1 = (x + half, y, z - half)
        v2 = (x + half, y, z + half)
        v3 = (x - half, y, z + half)
        v4 = (x, y + height, z) 

        self.tris = [
            Triangle(v4, v1, v0, color),
            Triangle(v4, v2, v1, color),
            Triangle(v4, v3, v2, color),
            Triangle(v4, v0, v3, color),
            Triangle(v0, v1, v2, color),
            Triangle(v0, v2, v3, color),
        ]

        self.vertex_normals = {v: [0.0, 0.0, 0.0] for v in [v0, v1, v2, v3, v4]}
        counts = {v: 0 for v in [v0, v1, v2, v3, v4]}

        for tri in self.tris:
            for vert in [tri.v0, tri.v1, tri.v2]:
                self.vertex_normals[vert] = add(self.vertex_normals[vert], tri.normal)
                counts[vert] += 1

        for vert in self.vertex_normals:
            self.vertex_normals[vert] = normalize(mul(self.vertex_normals[vert], 1.0 / counts[vert]))

    def intersect(self, ray_origin, ray_direction):
        closest_t = None
        hit_tri = None
        for tri in self.tris:
            t = tri.intersect(ray_origin, ray_direction)
            if t is not None and t > E and (closest_t is None or t < closest_t):
                closest_t = t
                hit_tri = tri
        if hit_tri is None:
            return None
        return closest_t, hit_tri

    def get_normal(self, hit_point, triangle):
        u, v, w = triangle.barycentric_coords(hit_point)
        n0 = self.vertex_normals[triangle.v0]
        n1 = self.vertex_normals[triangle.v1]
        n2 = self.vertex_normals[triangle.v2]
        interpolated = normalize(add(add(mul(n0, u), mul(n1, v)), mul(n2, w)))
        return interpolated


# Object definition of a Triangle
class Triangle:
    def __init__(self, v0, v1, v2, color=(200,200,200)):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.color = color

        edge1 = subtract(v1, v0)
        edge2 = subtract(v2, v0)
        self.normal = normalize(cross(edge1, edge2))

    def intersect(self, ray_origin, ray_direction):
        edge1 = subtract(self.v1, self.v0)
        edge2 = subtract(self.v2, self.v0)
        h = cross(ray_direction, edge2)
        a = dot(edge1, h)
        if abs(a) < E:
            return None

        f = 1.0 / a
        s = subtract(ray_origin, self.v0)
        u = f * dot(s, h)
        if u < 0.0 or u > 1.0:
            return None

        q = cross(s, edge1)
        v = f * dot(ray_direction, q)
        if v < 0.0 or u + v > 1.0:
            return None

        t = f * dot(edge2, q)
        return t if t > E else None
    
    def barycentric_coords(self, p):
        a, b, c = self.v0, self.v1, self.v2
        v0 = subtract(b, a)
        v1 = subtract(c, a)
        v2 = subtract(p, a)
        d00 = dot(v0, v0)
        d01 = dot(v0, v1)
        d11 = dot(v1, v1)
        d20 = dot(v2, v0)
        d21 = dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w
        return u, v, w

    def get_normal(self, hit_point=None):
        return self.normal


# Object definition of a Cube
class Cube:
    def __init__(self, center, size, color=(200,200,200)):
        half = size / 2.0
        x, y, z = center

        self.tris = [
            Triangle((x-half,y-half,z+half), (x+half,y-half,z+half), (x+half,y+half,z+half), color),
            Triangle((x-half,y-half,z+half), (x+half,y+half,z+half), (x-half,y+half,z+half), color),

            Triangle((x-half,y-half,z-half), (x+half,y+half,z-half), (x+half,y-half,z-half), color),
            Triangle((x-half,y-half,z-half), (x-half,y+half,z-half), (x+half,y+half,z-half), color),

            Triangle((x-half,y-half,z-half), (x-half,y-half,z+half), (x-half,y+half,z+half), color),
            Triangle((x-half,y-half,z-half), (x-half,y+half,z+half), (x-half,y+half,z-half), color),

            Triangle((x+half,y-half,z-half), (x+half,y+half,z+half), (x+half,y-half,z+half), color),
            Triangle((x+half,y-half,z-half), (x+half,y+half,z-half), (x+half,y+half,z+half), color),

            Triangle((x-half,y+half,z-half), (x-half,y+half,z+half), (x+half,y+half,z+half), color),
            Triangle((x-half,y+half,z-half), (x+half,y+half,z+half), (x+half,y+half,z-half), color),

            Triangle((x-half,y-half,z-half), (x+half,y-half,z+half), (x-half,y-half,z+half), color),
            Triangle((x-half,y-half,z-half), (x+half,y-half,z-half), (x+half,y-half,z+half), color),
        ]

    def intersect(self, ray_origin, ray_direction):
        closest_t = None
        hit_tri = None
        for tri in self.tris:
            t = tri.intersect(ray_origin, ray_direction)
            if t is not None and (closest_t is None or t < closest_t):
                closest_t = t
                hit_tri = tri
        if hit_tri is None:
            return None
        return closest_t, hit_tri

    def get_normal(self, hit_point, triangle):
        return triangle.normal

