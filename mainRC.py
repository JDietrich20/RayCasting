"""
Julianna Dietrich
Ray Casting Assignment 

* <= 3 spheres + 1 plane
* Point light source (intensity and location)
* Eye position
* Scene configuration
* Surface properties (k value in lighting model): Diffuse, Ambient, Specular
* View plane size and location 
* Shadow rays 
* Anti-aliasing (supersampling)

Extra:
(X) Texture Mapping 
(X) Add other objects like ellipsoids, triangles, boxes, etc. 
(X) Ray Tracing with Reflections and Refractions
"""

from PIL import Image
from objects import Sphere, Plane, Cube, Pyramid
from vec3 import E, subtract, dot, mul, normalize
import math

IMAGE_W, IMAGE_H = 600, 600 # Image dimensions

# Camera/Eye Location
EYE = (0.0, 1.0, -4.0)  # Ex, Ey, Ez
VIEW_SIZE = 1.0  # size fo view plane in world units (controls FOV width)

# Light location and intensity
LIGHT_POS = (5.0, 10.0, -5.0)  # Lx, Ly, Lz
LIGHT_INTENSITY = (1.0, 1.0, 1.0)  # R, G, B

# Material coefficients
Kd = 0.8  # diffuse
Ka = 0.2  # ambient
Ks = 0.5  #specular
alpha = 50 

#Texture mapping
tex = Image.open("tex/woodTexture.png").convert("RGB")
texturePixels = tex.load()
textureWidth, textureHeight = tex.size

# Anti-aliasing (3x3 = 9 rays per pixel)
AA_SAMPLES = 1

# Scene setup: spheres, plane, cube, pyramid
scene = [
    Sphere((0.0, 1.0, 5.0), 1.0, (255, 0, 0)), # Red Sphere located in the center 
    Cube((2.5, 1.0, 7.0), 2.0, (0, 0, 255)), # Blue Cube to the right
    Pyramid(( -2.0, 0.0, 4.0), 1.5, 2.0, (0, 255, 0)), # Green Pyramid to the left
    Plane( 
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        color=(255,255,255),
        texture=tex, #Giving the plane a texture of a wooden floor
        uv_scale=0.5 
    )
]

# Convert 3D point to UV coordinates and get texture color
def plane_uv(hit_point, scale=1.0):
    x, y, z = hit_point # Extracting 3D coordinates from where the ray hit the plane
    # Mapping x and z coordinates (3D world coordinates) to u and v texture coordinates
    u = (x * scale) % 1.0
    if u < 0: u += 1.0 # Ensuring u is positive
    v = (z * scale) % 1.0

    # Converting u,v to texture pixel coordinates
    tx = int(u * textureWidth) % textureWidth
    ty = int(v * textureHeight) % textureHeight

    return texturePixels[tx, ty] #returning the color from the texture image

# Lighting using the Phong reflection model
def compute_lighting(point, normal, base_color, view_dir):
    # Ambient light
    r = base_color[0] * Ka
    g = base_color[1] * Ka
    b = base_color[2] * Ka

    # Computing light direction
    L = normalize((LIGHT_POS[0]-point[0], LIGHT_POS[1]-point[1], LIGHT_POS[2]-point[2]))
    
    # Diffuse light
    n_dot_l = max(dot(normal, L), 0.0)
    r += base_color[0] * Kd * n_dot_l * LIGHT_INTENSITY[0]
    g += base_color[1] * Kd * n_dot_l * LIGHT_INTENSITY[1]
    b += base_color[2] * Kd * n_dot_l * LIGHT_INTENSITY[2]

    # Specular light
    R = subtract(mul(normal, 2 * dot(normal, L)), L) 
    r_dot_v = max(dot(R, view_dir), 0.0)
    spec = Ks * (r_dot_v ** alpha)
    r += base_color[0] * spec * LIGHT_INTENSITY[0]
    g += base_color[1] * spec * LIGHT_INTENSITY[1]
    b += base_color[2] * spec * LIGHT_INTENSITY[2]

    # Keeps color values within valid range
    return (int(min(r, 255)), int(min(g, 255)), int(min(b, 255)))


def trace_scene(ray_origin, ray_direction):
    closest_t = None
    hit_obj = None
    hit_extra = None

    for obj in scene:
        res = obj.intersect(ray_origin, ray_direction)
        if res is None:
            continue
        if isinstance(res, tuple):
            t, extra = res
        else:
            t = res
            extra = None
        if t is not None and t > E and (closest_t is None or t < closest_t):
            closest_t = t
            hit_obj = obj
            hit_extra = extra

    if closest_t is None:
        return None

    hit_point = (ray_origin[0] + closest_t * ray_direction[0],
                 ray_origin[1] + closest_t * ray_direction[1],
                 ray_origin[2] + closest_t * ray_direction[2])

    if isinstance(hit_obj, (Cube, Pyramid)):
        normal = hit_obj.get_normal(hit_point, hit_extra)
        color = hit_extra.color
    else:
        normal = hit_obj.get_normal(hit_point)
        color = hit_obj.color

    return {
        "t": closest_t,
        "object": hit_obj,
        "hit_point": hit_point,
        "normal": normal,
        "color": color,
        "extra": hit_extra
    }

def is_in_shadow(point, light_pos):
    to_light = (light_pos[0]-point[0], light_pos[1]-point[1], light_pos[2]-point[2])
    dist_to_light = math.sqrt(to_light[0]**2 + to_light[1]**2 + to_light[2]**2)
    dir_to_light = normalize(to_light)
    shadow_origin = (point[0] + dir_to_light[0]*E, point[1] + dir_to_light[1]*E, point[2] + dir_to_light[2]*E)
    for obj in scene:
        res = obj.intersect(shadow_origin, dir_to_light)
        if res is None:
            continue
        if isinstance(res, tuple):
            t, _ = res
        else:
            t = res
        if t and t > E and t < dist_to_light:
            return True
    return False


# Rendering loop (Ray Casting Algorithm)
image = Image.new("RGB", (IMAGE_W, IMAGE_H))
pixels = image.load()

for px in range(IMAGE_W):
    for py in range(IMAGE_H):
        r_acc = g_acc = b_acc = 0

        for i in range(AA_SAMPLES):
            for j in range(AA_SAMPLES):
                dx = (px + (i + 0.5)/AA_SAMPLES) / IMAGE_W - 0.5
                dy = (py + (j + 0.5)/AA_SAMPLES) / IMAGE_H - 0.5

                x = EYE[0] + dx * VIEW_SIZE
                y = EYE[1] - dy * VIEW_SIZE
                z = EYE[2] + VIEW_SIZE
                direction = normalize((x - EYE[0], y - EYE[1], z - EYE[2]))

                hit = trace_scene(EYE, direction)

                if hit is None:
                    color = (0, 0, 0) #black color
                else:
                    hit_obj = hit["object"]
                    hit_extra = hit["extra"] 

                    if hasattr(hit_obj, "texture") and hit_obj.texture is not None:
                        if isinstance(hit_obj, Plane):
                            base_color = plane_uv(hit["hit_point"])
                        elif isinstance(hit_obj, (Cube, Pyramid)) and hit_extra is not None:
                            base_color = hit_extra.color
                        else:
                            base_color = hit_extra.color if hit_extra is not None else (255,255,255)
                    else:
                        if isinstance(hit_obj, (Cube, Pyramid)) and hit_extra is not None:
                            base_color = hit_extra.color
                        else:
                            base_color = hit_obj.color

                    if is_in_shadow(hit["hit_point"], LIGHT_POS):
                        color = (int(base_color[0] * Ka),
                                 int(base_color[1] * Ka),
                                 int(base_color[2] * Ka))
                    else:
                        view_dir = normalize(subtract(EYE, hit["hit_point"]))
                        color = compute_lighting(hit["hit_point"], hit["normal"], base_color, view_dir)

                r_acc += color[0]
                g_acc += color[1]
                b_acc += color[2]
          

        scale = AA_SAMPLES * AA_SAMPLES

        pixels[px, py] = (int(r_acc/scale), int(g_acc/scale), int(b_acc/scale))

    # Progress indicator
    if px % 40 == 0:
        print(f"Progress: {px}/{IMAGE_W}")

print("Rendering finished.")
image.show()
image.save("Hw3_RAYCAST_RESULT.png")

