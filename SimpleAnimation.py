'''
Display an animation from a sequence of frames in a pygame window.

TODO: dump animation to video file (with opencv?)
'''

import pygame


# TODO: allow up/downsampling to change window size?
def animate(aspect_ratio, frame_iterator, frame_rate=10):
    '''
    Display a series of 3D numpy arrays as RGB frames in a pygame window.
    
    Parameters
    ----------
    aspect_ratio : tuple
        (max_u, max_v)
    frame_iterator : iterator
        yields RGB frames contained in ndarrays of shape (max_u, max_v, 3)
    frame_rate : int, optional
        The default is 10.
    '''
    pygame.init()
    screen = pygame.display.set_mode(aspect_ratio)
    clock = pygame.time.Clock()
    for frame in frame_iterator:
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        clock.tick(frame_rate)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        

# simplistic example animation
if __name__ == '__main__':
    import numpy as np
    from numpy import random
    
    def fuzzyFadeIn(aspect_ratio, p):
        max_u, max_v = aspect_ratio
        frame = np.zeros((max_u, max_v, 3))
        while not frame.all():
            yield frame
            sample = random.rand(max_u, max_v)
            frame[sample < p] = 255
    
    aspect_ratio = (640, 480)
    p = 0.5
    animate(aspect_ratio, fuzzyFadeIn(aspect_ratio, p))
            
        
        