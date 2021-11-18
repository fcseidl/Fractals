'''
Uses the PixelatedFractals module to animate fractal zooms as sequences of 
numpy arrays, which can be visualized using the SimpleAnimation module.

TODO: simultaneous pan and zoom
TODO: fix jitter from rounding aspect ratio

way hard:
TODO: allow exploration of fractal in real time with scroll bar and wasd
'''

import numpy as np
from skimage.transform import rescale
from simple_animation import animate
from pixelated_fractals import PixelatedFractal
        

# fill a frame with a pixelated fractal from the top
def fillFromTop(frame, pixelated_fractal, thickness=1):
    u_max, v_max = pixelated_fractal.aspect_ratio
    yield frame
    for v in range(0, v_max, thickness):
        for u in range(u_max):
            for dv in range(min(thickness, v_max - v)):
                frame[u, v+dv] = pixelated_fractal.pixelColor(u, v+dv)
        yield frame
    return frame


def smoothZoom(pixelated_fractal, 
               zoom_point, 
               doubling_frames=98,
               _sf=np.sqrt(np.e)):
    """
    Continuous zoom towards a fixed point, doubling resolution every fixed 
    number of frames. Achieved by interpolating a sequence of images at 
    resolutions increasing by a factor of _sf, which by default takes the 
    optimal value sqrt(e).
    """
    # center pixelated_fractal on target point and increase resolution
    pixelated_fractal.adjustZoom(1.0, zoom_point)
    u_max, v_max = pixelated_fractal.aspect_ratio
    pixelated_fractal.adjustResolution(_sf)
    # copy fractal object will be used to precompute image at next resolution
    subpixel_fractal = pixelated_fractal.deepcopy()
    subpixel_frame = 255 * np.ones(subpixel_fractal.aspect_ratio + (3,))
    # fill from top while preparing subpixel frame for zoom
    for subpixel_frame in fillFromTop(subpixel_frame, 
                                      subpixel_fractal, 
                                      thickness=20):
        yield rescale(subpixel_frame, (1/_sf, 1/_sf, 1))
        
    # constants
    sub_u_max, sub_v_max = subpixel_fractal.aspect_ratio
    sub_v_grid, sub_u_grid = np.meshgrid(
        np.arange(sub_v_max), 
        np.arange(sub_u_max)
        )
    zoom_per_frame = 2 ** (1 / doubling_frames)
    sf_frames = int(np.log(_sf) / np.log(zoom_per_frame))
    
    # zoom in while preparing a subimage at higher resolution
    while True:
        northwest = subpixel_fractal.pixelSpaceToC(0, 0) - zoom_point
        southeast = subpixel_fractal.pixelSpaceToC(sub_u_max - 1, sub_v_max - 1) - zoom_point
        new_subpixel_frame = np.zeros_like(subpixel_frame)
        subpixel_fractal.adjustZoom(_sf, zoom_point)
        orbit_grid = subpixel_fractal.pixelSpaceToC(sub_u_grid, sub_v_grid)
        holomorphic_map = subpixel_fractal.get_map(orbit_grid)
        iteration = 0
        
        for n in range(sf_frames):
            # compute coordinates of new corners in subpixel resolution
            u_west, v_north = pixelated_fractal.cToPixelSpace(
                zoom_point + (northwest / zoom_per_frame**n)
            )
            u_east, v_south = pixelated_fractal.cToPixelSpace(
                zoom_point + (southeast / zoom_per_frame**n)
            )
            
            yield rescale(
                    subpixel_frame[u_west:u_east, v_north:v_south],
                    (zoom_per_frame**n / _sf, zoom_per_frame**n / _sf, 1)
                )
            
            # compute next few iterations of the map at every pixel
            stop_iter = int(pixelated_fractal.max_iter * (n + 1) / sf_frames)
            for it in range(iteration, stop_iter):
                new_subpixel_frame[
                    (new_subpixel_frame.sum(axis=2) == 0) 
                    * (np.abs(orbit_grid) > 2)
                    ] = pixelated_fractal.colorFromTime(it)
                orbit_grid = holomorphic_map(orbit_grid)
            iteration = stop_iter
        
        subpixel_frame = new_subpixel_frame
        pixelated_fractal.adjustZoom(_sf, zoom_point)
            


# example zoom
if __name__ == '__main__':
    from numpy import random
    seed = random.randint(1000000)
    
    #print('random seed =', seed)
    random.seed(seed)
    
    aspect_ratio = (640, 480)
    exponent = 2
    
    # quadratically interpolate color cycle
    N = 11
    color_cycle = [np.array([255, 0, 0]) * (1 - n/N)**2 
                   + np.array([0, 0, 255]) * (n/N)**2
                   for n in range(N+1)]
    
    fractal = PixelatedFractal(aspect_ratio, 
                                  exponent=exponent,
                                  julia_param=0-0.8j,
                                  color_cycle=color_cycle,
                                  max_iter=80)
    # zoom toward Feigenbaum point looks periodic
    frame_iterator = smoothZoom(fractal, 
                                doubling_frames=64, 
                                zoom_point=0
                                )
    animate(aspect_ratio, 
            frame_iterator, 
            fps=32,
            title='fractalZoom',
            video_dump=1)
    