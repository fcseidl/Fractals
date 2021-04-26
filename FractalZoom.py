'''
Uses the PixelatedFractals module to animate fractal zooms as sequences of 
numpy arrays, in the input format of the SimpleAnimation module.
'''

import numpy as np
from SimpleAnimation import animate
from PixelatedFractals import MandelbrotGreyscale


RED = np.array([255, 0, 0])


# AUXILIARY FUNCTIONS

# TODO: this might need to be faster/smoother
# create array of different size with nearest-neighbor tiling
def resample(frame, new_aspect):
    w_old, h_old, _ = frame.shape
    w_new, h_new = new_aspect
    w_scale = w_old / w_new
    h_scale = h_old / h_new
    result = np.empty((w_new, h_new, 3))
    for i_new in range(w_new):
        i_old = int(w_scale * i_new)
        for j_new in range(h_new):
            j_old = int(h_scale * j_new)
            result[i_new, j_new] = frame[i_old, j_old]
    return result

# TODO: this sucks, find a better way to do it!
# find a point on the border of the fractal
def findLeftBorder(frame):
    u_max, v_max, _ = frame.shape
    for u in range(u_max):
        for v in range(v_max):
            if (frame[u, v] == 0).all():
                return u, v


# FRAME SEQUENCE GENERATORS        

def fillFromTop(frame, pixelated_fractal, thickness=1):
    u_max, v_max = pixelated_fractal.aspect_ratio
    yield frame
    for v in range(0, v_max, thickness):
        for u in range(u_max):
            for dv in range(thickness):
                frame[u, v+dv] = pixelated_fractal.color(u, v+dv)
        '''if v < v_max - thickness:
            frame[:, v + thickness] = RED'''
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
    '''# draw rectangle
    if u_left >= 0:      # left side
        frame[u_left, max(v_top,0):min(v_top+box_height,v_max)] = RED
    if v_top >= 0:       # top side
        frame[max(u_left,0):min(u_left+box_width,u_max), v_top] = RED
    if u_left + box_width < u_max:      # right side
        frame[u_left+box_width-1, max(v_top,0):min(v_top+box_height,v_max)] = RED
    if v_top + box_height < v_max:      # bottom side
        frame[max(u_left,0):min(u_left+box_width,u_max), v_top+box_height-1] = RED'''
    # expand rectangle to fill frame
    try:
        new_frame = frame[u_left:u_left+box_width, v_top:v_top+box_height].copy()
        for t in range(duration + 1):
            u_t = int((duration - t) * u_left / duration)
            v_t = int((duration - t) * v_top / duration)
            w_t = int((t * u_max + (duration - t) * box_width) / duration)
            h_t = int((t * v_max + (duration - t) * box_height) / duration)
            frame[u_t:u_t+w_t, v_t:v_t+h_t] \
                = resample(new_frame, (w_t, h_t))
            yield frame
    except IndexError:
        raise ValueError('subregion to magnify must be contained in current frame')
    return frame
     
def fractalZoom(pixelated_fractal, scale_factor=4):
    u_max, v_max = pixelated_fractal.aspect_ratio
    frame = 255 * np.ones((u_max, v_max, 3))
    zoom_level = 1
    while True:
        print('Zoom level %f' % zoom_level)
        frame = yield from fillFromTop(frame, pixelated_fractal, thickness=10)
        u, v = findLeftBorder(frame)
        frame = yield from magnifySubframe(u, v, frame, scale_factor, duration=30)
        new_window_center = pixelated_fractal.pixelSpaceToC(u, v)
        pixelated_fractal.adjustZoom(scale_factor, new_window_center)
        zoom_level *= scale_factor
    

# example zoom
if __name__ == '__main__':
    aspect_ratio = (960, 720)
    pixelated_fractal = MandelbrotGreyscale(aspect_ratio,
                                            exponent=2, 
                                            julia_param=0.25,
                                            max_iter=40)
    frame_iterator = fractalZoom(pixelated_fractal)
    animate(aspect_ratio, frame_iterator, frame_rate=50)
    