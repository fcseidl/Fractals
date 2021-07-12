"""
Interactive visualization of a 2D square in 3D space.
"""

import numpy as np
from numpy.linalg import norm
from pygame.constants import K_DOWN, K_RIGHT, K_UP, K_LEFT
from sklearn.neighbors import KDTree
import pygame


mesh = 1e-2
ang_speed = 1                   # radians / sec
fps = 10

aspect = (200, 150)
render_dist = 4
field_of_view = np.pi / 2       # horizontally
vert_fov = field_of_view * aspect[1] / aspect[0]
background = np.array([255, 255, 255])
scene_color = np.zeros(3)

# dimensions of viewport; camera at origin looking down x-axis
halfwidth = np.tan(field_of_view / 2)
halfheight = halfwidth * aspect[1] / aspect[0]

# densely sample points of square into kdtree
X = Y = np.linspace(-1, 1, num=int(2/mesh))
scene = np.array([
    [x, y, 0] for x in X for y in Y
])
tree = KDTree(scene)

# initialize animation at (2, 0, 0)
pygame.init()
window = pygame.display.set_mode(aspect)
pygame.display.set_caption('square')
clock = pygame.time.Clock()
r_asc = dec = 0


# auxiliary functions implementing ray marching

# returns 3D position of camera on radius 2 ball around origin
def camera_location():
    x = np.cos(r_asc) * np.sin(dec)
    y = np.sin(r_asc) * np.sin(dec)
    z = np.cos(dec)
    return 2 * np.array([x, y, z])

# returns unit vector in the direction of a pixel's center ray
def direction(u, v, cam_pos):
    # direction in camera coordinates, with cam looking down x-axis
    cam_relative = np.array([
        1.,
        (2 * (u / aspect[0]) * halfwidth - halfwidth),
        (2 * (v / aspect[1]) * halfheight - halfheight)
    ])
    cam_relative /= norm(cam_relative)
    # now transform to absolute coords
    xaxis = -cam_pos / norm(cam_pos)
    yaxis = np.cross(xaxis, np.array([0,0,1]))
    zaxis = np.cross(yaxis, xaxis)
    tmat = np.linalg.pinv(np.array([xaxis, yaxis, zaxis]))
    return cam_relative.dot(tmat)

# march a point from start in the direction of a unit vector
# until hitting the scene or passing render distance.
# Return color of the point where the ray hits the scene
# TODO: glow
def ray_march(start, dir):
    point = start.copy()
    while norm(point - start) < render_dist:
        query_result = tree.query(np.array([point]))
        dist = query_result[0][0, 0]
        if dist < mesh:
            return scene_color
        point += dist * dir

    return background

def pixel_color(u, v):
    start = camera_location()
    dir = direction(u, v, start)
    pix = ray_march(start, dir)
    return pix


# main loop
while True:
    keys = pygame.key.get_pressed()
    if keys[K_UP]:
        r_asc += fps * ang_speed
    if keys[K_LEFT]:
        dec += fps * ang_speed
    if keys[K_DOWN]:
        r_asc -= fps * ang_speed
    if keys[K_RIGHT]:
        dec -= fps * ang_speed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    frame = np.empty(aspect + (3,))
    for u in range(aspect[0]):
        print('column', u)
        for v in range(aspect[1]):
            frame[u, v] = pixel_color(u, v)

    surf = pygame.surfarray.make_surface(frame)
    window.blit(surf, (0,0))
    clock.tick(fps)
    pygame.display.update()