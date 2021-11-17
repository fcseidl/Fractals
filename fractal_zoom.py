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

# Zoom to region of frame centered on u, v with specified scale.
# Note that the region must be entirely contained in the frame!
def magnifySubframe(u, v, frame, scale_factor, duration):
    # compute new window specification
    u_max, v_max, _ = frame.shape
    box_width = int(u_max / scale_factor)
    box_height = int(v_max / scale_factor)
    u_left = u - int(box_width / 2)
    v_top = v - int(box_height / 2)
    # expand rectangle to fill frame
    try:
        magnify = frame[u_left:u_left+box_width, v_top:v_top+box_height].copy()
        for t in range(duration + 1):
            u_t = int((duration - t) * u_left / duration)
            v_t = int((duration - t) * v_top / duration)
            w_t = int((t * u_max + (duration - t) * box_width) / duration)
            h_t = int((t * v_max + (duration - t) * box_height) / duration)
            frame[u_t:u_t+w_t, v_t:v_t+h_t] = rescale(magnify, (w_t/box_width, h_t/box_height, 1))
            yield frame
    except IndexError:
        raise ValueError('subregion to magnify must be contained in current frame')
     
def uglyZoom(pixelated_fractal, zoom_point, scale_factor=4):
    """Repeatedly zoom and enhance, with noticeable discrete steps."""
    u_max, v_max = pixelated_fractal.aspect_ratio
    frame = 255 * np.ones((u_max, v_max, 3))  #, dtype='uint8') for some reason this breaks rescaling
    zoom_level = 1
    while True:
        print('Zoom level %f' % zoom_level)
        frame = yield from fillFromTop(frame, pixelated_fractal, thickness=10)
        u, v = pixelated_fractal.cToPixelSpace(zoom_point)
        yield from magnifySubframe(u, v, frame, scale_factor, duration=20)
        new_window_center = pixelated_fractal.pixelSpaceToC(u, v)
        pixelated_fractal.adjustZoom(scale_factor, new_window_center)
        zoom_level *= scale_factor


def smoothZoom(pixelated_fractal, zoom_point, doubling_frames=256, _sf=np.sqrt(np.e)):
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
    zoom_per_frame = 2 ** (1 / doubling_frames)
    sf_frames = int(np.log(_sf) / np.log(zoom_per_frame))
    
    # zoom in while preparing a subimage at higher resolution
    while True:
        northwest = subpixel_fractal.pixelSpaceToC(0, 0) - zoom_point
        southeast = subpixel_fractal.pixelSpaceToC(sub_u_max - 1, sub_v_max - 1) - zoom_point
        new_subpixel_frame = np.empty_like(subpixel_frame)
        subpixel_fractal.adjustZoom(_sf, zoom_point)
        
        v_filled = 0
        for n in range(sf_frames):
            # compute coordinates of new corners in subpixel resolution
            u_west, v_north = pixelated_fractal.cToPixelSpace(zoom_point + (northwest / zoom_per_frame**n))
            u_east, v_south = pixelated_fractal.cToPixelSpace(zoom_point + (southeast / zoom_per_frame**n))
            yield rescale(
                subpixel_frame[u_west:u_east, v_north:v_south],
                (zoom_per_frame**n / _sf, zoom_per_frame**n / _sf, 1)
                )
            
            # compute more of the new subpixel frame
            fill_to = int((n + 1) * sub_v_max / sf_frames)
            for v in range(v_filled, fill_to):
                for u in range(sub_u_max):
                    new_subpixel_frame[u, v] = subpixel_fractal.pixelColor(u, v)
            v_filled = fill_to
        
        subpixel_frame = new_subpixel_frame
        pixelated_fractal.adjustZoom(_sf, zoom_point)
            
            
            
    


# example zoom
if __name__ == '__main__':
    from numpy import random
    seed = random.randint(1000000)
    
    print('random seed =', seed)
    random.seed(seed)
    
    aspect_ratio = (640, 480)
    exponent = 2
    
    # quadratically interpolate color cycle
    N = 11
    color_cycle = [np.array([255, 0, 0]) * (1 - n/N)**2 
                   + np.array([0, 0, 255]) * (n/N)**2
                   for n in range(N+1)]
    
    mandelbrot = PixelatedFractal(aspect_ratio, 
                                  exponent=exponent,
                                  color_cycle=color_cycle,
                                  max_iter=60)
    # zoom toward Feigenbaum point looks periodic
    '''frame_iterator = uglyZoom(mandelbrot, 
                              zoom_point=-1.401155189+0j, 
                              scale_factor=4.66920109)'''
    frame_iterator = smoothZoom(mandelbrot, 
                                doubling_frames=98, 
                                zoom_point=-2#-1.401155189+0j
                                )
    animate(aspect_ratio, 
            frame_iterator, 
            fps=32,
            title='fractalZoom',
            video_dump=True)
    