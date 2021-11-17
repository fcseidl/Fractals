'''
Show orbits of points on top of a fractal.

TODO: allow zoom!
TODO: precision is pretty bad
'''

import pygame
import numpy as np
from pixelated_fractals import PixelatedFractal


# set parameters
frame_rate = 10
orbit_length = 50
orbit_color = (0, 255, 0)
aspect_ratio = (1280, 800)

# quadratically interpolate color cycle. 
# note pygame orders RGB differently from opencv
N = 11
color_cycle = [np.array([0, 0, 255]) * (1 - n/N)**2 
               + np.array([255, 0, 0]) * (n/N)**2
               for n in range(N+1)]

fractal = PixelatedFractal(aspect_ratio, color_cycle=color_cycle)

# use precomputed previous background or recompute?
if 1:
    background = np.load('background.npy')
else:
    print('computing background frame...')
    background = np.zeros((aspect_ratio[0], aspect_ratio[1], 3))
    for u in range(aspect_ratio[0]):
        for v in range(aspect_ratio[1]):
            background[u, v] = fractal.pixelColor(u, v)
    np.save('background', background)

# initialize display window
pygame.init()
screen = pygame.display.set_mode(aspect_ratio)
clock = pygame.time.Clock()
orbit = None

# main loop
quitting = False
while not quitting:
    # draw background
    surf = pygame.surfarray.make_surface(background)
    
    # draw orbit
    if orbit is not None:
        pygame.draw.lines(surf, orbit_color, closed=False, points=orbit)
    
    # show new frame
    screen.blit(surf, (0, 0))
    clock.tick(frame_rate)
    pygame.display.update()
    
    # handle events
    for event in pygame.event.get():
        # click on a point to see orbit
        if event.type == pygame.MOUSEBUTTONDOWN:
            u, v = pygame.mouse.get_pos()
            orbit = fractal.orbit_pixels(u, v, orbit_length)   # TODO: implement this
        if event.type == pygame.QUIT:
            pygame.quit()
            quitting = True
            