"""
Interactive visualization of a 2D square in 3D space.
"""

import numpy as np
from pygame.constants import K_DOWN, K_RIGHT, K_UP, K_LEFT
from sklearn.neighbors import KDTree
import pygame


mesh = 1e-2
ang_speed = 1                   # radians / sec
fps = 10

aspect = (640, 480)
field_of_view = np.pi / 2       # horizontally


# densely sample points of square into kdtree
X = Y = np.linspace(-1, 1, num=int(2/mesh))
points = np.array([
    [x, y, 0] for x in X for y in Y
])
tree = KDTree(points)


# initialize animation at (2, 0, 0)
pygame.init()
window = pygame.display.set_mode(aspect)
pygame.display.set_caption('square')
clock = pygame.time.Clock()
r_asc = dec = 0


# auxiliary functions implementing ray marching

def camera_location():
    x = np.cos(r_asc) * np.sin(dec)
    y = np.sin(r_asc) * np.sin(dec)
    z = np.cos(dec)
    return 2 * np.array([x, y, z])

def direction(u, v, start):
    pass

def ray_march(start, dir):
    pass

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
        for v in range(aspect[1]):
            frame[u, v] = pixel_color(u, v)

    clock.tick(fps)
    pygame.display.update()