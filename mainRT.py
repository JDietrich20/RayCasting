"""
Julianna Dietrich
Ray Casting Assignment (Ray Tracing Extension)

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
MAX_DEPTH = 3 # Max recursion depth for reflections/refractions

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

# Anti-aliasing (3x3 = 9 rays per pixel)
AA_SAMPLES = 3

def set_materials_for_scene():
    for obj in scene:
        obj.kr = 0.0
        obj.kt = 0.0
        obj.ior = 1.0
        obj.ka = Ka
        obj.kd = Kd

    # Red sphere reflective
    for obj in scene:
        if isinstance(obj, Sphere) and obj.color == (255, 0, 0):
            obj.kr = 0.8
            obj.kt = 0.0

    # Green sphere refractive/glass
    for obj in scene:
        if isinstance(obj, Sphere) and obj.color == (255, 255, 255):
            obj.kr = 0.05
            obj.kt = 0.99 
            obj.ior = 1.33 
            obj.kd = 0.2 
            obj.ka = 0.1
            obj.color = (200, 200, 255)

    for obj in scene:
        if isinstance(obj, Plane):
            obj.kr = 0.0
            obj.kt = 0.0


# Scene setup: spheres, plane, cube, pyramid
scene = [
    Sphere((0.0, 1.0, 5.0), 1.0, (255, 0, 0)),
    Cube((2.5, 1.0, 7.0), 2.0, (0, 0, 255)),
    Pyramid(( -2.0, 0.0, 4.0), 1.5, 2.0, (0, 255, 0)),
    Sphere((1.0, 1.0, 2.0), 1.0, (255, 255, 255)),
    Plane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        color=(255,255,255),
        texture=tex,
        uv_scale=0.5 
    )
]

set_materials_for_scene()

def compute_lighting(point, normal, base_color, view_dir):
    
    r = base_color[0] * Ka
    g = base_color[1] * Ka
    b = base_color[2] * Ka

    
    L = normalize((LIGHT_POS[0]-point[0], LIGHT_POS[1]-point[1], LIGHT_POS[2]-point[2]))
    n_dot_l = max(dot(normal, L), 0.0)

    r += base_color[0] * Kd * n_dot_l * LIGHT_INTENSITY[0]
    g += base_color[1] * Kd * n_dot_l * LIGHT_INTENSITY[1]
    b += base_color[2] * Kd * n_dot_l * LIGHT_INTENSITY[2]

    R = subtract(mul(normal, 2 * dot(normal, L)), L) 
    r_dot_v = max(dot(R, view_dir), 0.0)
    spec = Ks * (r_dot_v ** alpha)
    r += base_color[0] * spec * LIGHT_INTENSITY[0]
    g += base_color[1] * spec * LIGHT_INTENSITY[1]
    b += base_color[2] * spec * LIGHT_INTENSITY[2]


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


def reflect_dir(I, N):
    d = 2 * dot(I, N)
    return normalize((I[0] - d*N[0], I[1] - d*N[1], I[2] - d*N[2]))

def refract_dir(I, N, ior):
    cosi = max(-1.0, min(1.0, dot(I, N)))
    etai = 1.0
    etat = ior
    n = N
    if cosi < 0:
        cosi = -cosi
    else:
        n = (-N[0], -N[1], -N[2])
        etai, etat = etat, etai

    eta = etai / etat
    k = 1.0 - eta*eta * (1.0 - cosi*cosi)
    if k < 0:
        return None 
    refr = (
        eta * I[0] + (eta * cosi - math.sqrt(k)) * n[0],
        eta * I[1] + (eta * cosi - math.sqrt(k)) * n[1],
        eta * I[2] + (eta * cosi - math.sqrt(k)) * n[2]
    )
    return normalize(refr)

def get_base_color(hit):
    obj = hit["object"]
    pt = hit["hit_point"]
    extra = hit.get("extra", None)

    if extra is not None and hasattr(extra, "color"):
        base = extra.color
    elif hasattr(obj, "color") and obj.color is not None:
        base = obj.color
    else:
        base = (200, 200, 200)

    tex_img = None
    if extra is not None and hasattr(extra, "texture") and extra.texture is not None:
        tex_img = extra.texture
    elif hasattr(obj, "texture") and obj.texture is not None:
        tex_img = obj.texture

    if tex_img is not None:
        tex_pixels = tex_img.load()
        tw, th = tex_img.size

        if hasattr(obj, "center") and (getattr(obj, "radius", None) is not None):
            p_rel = subtract(pt, obj.center)
            p_norm = normalize(p_rel)
            theta = math.acos(max(-1.0, min(1.0, p_norm[1])))
            phi = math.atan2(p_norm[2], p_norm[0])
            u = (phi + math.pi) / (2 * math.pi)
            v = theta / math.pi
            tx = int(u * tw) % tw
            ty = int(v * th) % th
            return tex_pixels[tx, ty]

        if hasattr(obj, "normal") and hasattr(obj, "point"):
            x, y, z = pt
            scale = getattr(obj, "uv_scale", 1.0)
            u = (x * scale) % 1.0
            v = (z * scale) % 1.0
            tx = int(u * tw) % tw
            ty = int(v * th) % th
            return tex_pixels[tx, ty]

        if extra is not None and hasattr(extra, "v0"):
            if hasattr(extra, "tex_coords") and extra.tex_coords is not None:
                a, b, c = extra.v0, extra.v1, extra.v2
                u_b, v_b, w_b = barycentric_coords(pt, a, b, c)
                (u0, v0), (u1, v1), (u2, v2) = extra.tex_coords
                u = u0*u_b + u1*v_b + u2*w_b
                v = v0*u_b + v1*v_b + v2*w_b
                tx = int(u * tw) % tw
                ty = int(v * th) % th
                return tex_pixels[tx, ty]
            else:
                return base

    return base


def barycentric_coords(p, a, b, c):
    v0 = subtract(b, a)
    v1 = subtract(c, a)
    v2 = subtract(p, a)
    d00 = dot(v0, v0)
    d01 = dot(v0, v1)
    d11 = dot(v1, v1)
    d20 = dot(v2, v0)
    d21 = dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return (1/3, 1/3, 1/3)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return (u, v, w)

def blend_colors(c1, c2, weight):
    return (
        (1-weight)*c1[0] + weight*c2[0],
        (1-weight)*c1[1] + weight*c2[1],
        (1-weight)*c1[2] + weight*c2[2]
    )

def add_scaled_color(c1, c2, scale):
    return (
        c1[0] + c2[0]*scale,
        c1[1] + c2[1]*scale,
        c1[2] + c2[2]*scale
    )

def clamp_color(c):
    return (int(min(255, max(0, c[0]))),
            int(min(255, max(0, c[1]))),
            int(min(255, max(0, c[2]))))

def trace_ray(origin, direction, depth=0):
    if depth > MAX_DEPTH:
        return (0, 0, 0)

    hit = trace_scene(origin, direction)
    if hit is None:
        return (0, 0, 0)

    P = hit["hit_point"]
    N = hit["normal"]
    obj = hit["object"]

    kr = getattr(obj, "kr", 0.0)
    kt = getattr(obj, "kt", 0.0)
    ior = getattr(obj, "ior", 1.0)
    ka_obj = getattr(obj, "ka", Ka)
    kd_obj = getattr(obj, "kd", Kd)

    view_dir = normalize((-direction[0], -direction[1], -direction[2]))

    base_color = get_base_color(hit) or (200, 200, 200)

    # local illumination
    if is_in_shadow(P, LIGHT_POS):
        local_color = (base_color[0]*ka_obj,
                       base_color[1]*ka_obj,
                       base_color[2]*ka_obj)
    else:
        local_color = compute_lighting(P, N, base_color, view_dir)

    final_color = local_color

    # reflection
    if kr > 0.0 and depth < MAX_DEPTH:
        R = reflect_dir(direction, N)
        refl_origin = (P[0]+R[0]*E, P[1]+R[1]*E, P[2]+R[2]*E)
        refl_color = trace_ray(refl_origin, R, depth+1)
        final_color = add_scaled_color(final_color, refl_color, kr)

    # refraction
    if kt > 0.0 and depth < MAX_DEPTH:
        refr = refract_dir(direction, N, ior)
        if refr is None:
            R = reflect_dir(direction, N)
            refr_origin = (P[0]+R[0]*E, P[1]+R[1]*E, P[2]+R[2]*E)
            refr_color = trace_ray(refr_origin, R, depth+1)
        else:
            refr_origin = (P[0]+refr[0]*E, P[1]+refr[1]*E, P[2]+refr[2]*E)
            refr_color = trace_ray(refr_origin, refr, depth+1)

        tint_weight = 0.05
        tinted_refr = blend_colors(refr_color, base_color, tint_weight)
        final_color = blend_colors(final_color, tinted_refr, kt)

    return clamp_color(final_color)



# Rendering loop (Ray Tracing Algorithm)
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

                col = trace_ray(EYE, direction, depth=0)
                r_acc += col[0]
                g_acc += col[1]
                b_acc += col[2]
                

        scale = AA_SAMPLES * AA_SAMPLES
        pixels[px, py] = (int(r_acc/scale), int(g_acc/scale), int(b_acc/scale))

    # Progress indicator
    if px % 40 == 0:
        print(f"Progress: {px}/{IMAGE_W}")

print("Rendering finished.")
image.show()
image.save("Hw3_RAYTRACE_RESULT.png")

