'''
Uses the PixelatedFractals module to animate fractal zooms as sequences of 
numpy arrays, in the input format of the SimpleAnimation module.
'''

import numpy as np
from numpy import random
from SimpleAnimation import animate
from PixelatedFractal import MandelbrotGreyscale


RED = np.array([255, 0, 0])


# AUXILIARY FUNCTIONS
# TODO: make generators be classes with these as member functions?

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
                
# use border tracing algorithm to draw a pixelated fractal.
# NOTE: the fractal MUST BE CONNECTED!
class BorderTracer:
    
    NOT_COMPUTED = -1
    OUT_OF_BOUNDS = -2
    
    CLOCKWISE = {'N':'E', 'E':'S', 'S':'W', 'W':'N'}
    COUNTERCLOCKWISE = {'N':'W', 'W':'S', 'S':'E', 'E':'N'}
    
    def __init__(self, frame, pixelated_fractal, speed):
        self.frame = frame
        self.pixelated_fractal = pixelated_fractal
        self.speed = speed
        
        u_max, v_max = pixelated_fractal.aspect_ratio
        self.colors = pixelated_fractal.colors()
        self.trace = self.NOT_COMPUTED * np.ones((u_max, v_max), dtype=int)
        self.traced_idxs = np.zeros(len(self.colors))
    
    # return index of pixel color in self.colors, avoiding re-deciding pixels 
    # and bad indexing. Update frame when a new pixel is decided.
    def safeDynamicIndex(self, u, v):
        u_max, v_max = pixelated_fractal.aspect_ratio
        if u < 0 or v < 0 or u >= u_max or v >= v_max:
            return self.OUT_OF_BOUNDS
        if self.trace[u, v] == self.NOT_COMPUTED:
            c = self.pixelated_fractal.pixelColor(u, v)
            
            
            self.frame[u, v] = RED # TODO: return to c
            
            
            self.trace[u, v] = self.colors.index(c)
        return self.trace[u, v]
    
    # Trace along the border of the lemniscate starting at u, v.
    # Currently using square tracing.
    def traceLemniscate(self, u, v, interior_color_idx, direction, stop_at_edge):
        print('tracing lemniscate of color indexed by', interior_color_idx)
        step = 0
        u_start, v_start = u, v
        while True:
            # decide next step direction
            idx = self.safeDynamicIndex(u, v)
            if idx <= interior_color_idx:
                direction = self.CLOCKWISE[direction]
            else:
                direction = self.COUNTERCLOCKWISE[direction]
            # perform step
            if direction == 'N': 
                v -= 1
            elif direction == 'E':
                u += 1
            elif direction == 'S':
                v += 1
            elif direction == 'W':
                u -= 1
            # stop when loop is completed
            if u == u_start and v == v_start:
                print('finished tracing lemniscate: came full circle')
                break
            if idx == self.OUT_OF_BOUNDS and stop_at_edge:
                print('finished tracing lemniscate: went out of bounds')
                break
            # render progress
            if step % self.speed == 0:
                yield self.frame
            step += 1
        
    # trace edges of frame, stopping to trace lemniscates we cross
    # TODO: minimize code duplication
    def traceEdges(self):
        print('tracing edges')
        u_max, v_max = pixelated_fractal.aspect_ratio
        # top left corner
        prev = self.safeDynamicIndex(0, 0)
        # top
        for u in range(1, u_max):
            if u % self.speed == 0:
                yield self.frame
            current = self.safeDynamicIndex(u, 0)
            if current != prev:
                if current < prev:
                    yield from self.traceLemniscate(u, 0, current, 'E', stop_at_edge=True)
                    self.traced_idxs[current] = 1
                prev = current
        # right
        for v in range(1, v_max):
            if v % self.speed == 0:
                yield self.frame
            current = self.safeDynamicIndex(u_max-1, v)
            if current != prev:
                if current < prev:
                    yield from self.traceLemniscate(u_max-1, v, current, 'S', stop_at_edge=True)
                    self.traced_idxs[current] = 1
                prev = current
        # bottom
        for u in range(u_max-2, -1, -1):
            if u % self.speed == 0:
                yield self.frame
            current = self.safeDynamicIndex(u, v_max-1)
            if current != prev:
                if current < prev:
                    yield from self.traceLemniscate(u, v_max-1, current, 'W', stop_at_edge=True)
                    self.traced_idxs[current] = 1
                prev = current
        # left
        for v in range(v_max-2, -1, -1):
            if v % self.speed == 0:
                yield self.frame
            current = self.safeDynamicIndex(0, v)
            if current != prev:
                if current < prev:
                    yield from self.traceLemniscate(0, v, current, 'N', stop_at_edge=True)
                    self.traced_idxs[current] = 1
                prev = current
        # now we've traced the outermost lemnscate too
        self.traced_idxs[-1] = 1
        print('finished tracing edges')
    
    # search trace all untraced lemniscates crossing a column
    def traceCol(self, u):
        print('trace column', u)
        u_max, v_max = pixelated_fractal.aspect_ratio
        prev = self.safeDynamicIndex(u, 0)
        for v in range(1, v_max):
            if v % self.speed == 0:
                yield self.frame
            current = self.safeDynamicIndex(u, v)
            if current != prev:
                for idx in range(current, prev):
                    if not self.traced_idxs[idx]:
                        yield from self.traceLemniscate(u, v, idx, 'S', stop_at_edge=False)
                        self.traced_idxs[idx] = 1
                prev = current
        print('finished tracing column')
        
    # choose a random point in frame from the set by rejection sampling
    def chooseFromFractal(self):
        u_max, v_max = pixelated_fractal.aspect_ratio
        u, v = (0, 0)
        while self.safeDynamicIndex(u, v) > 0:
            u = random.randint(u_max)
            v = random.randint(v_max)
        return u, v
    
    # animate tracing all borders
    def generate(self):
        print('need to trace colors indexed by', 
              np.argwhere(self.traced_idxs - 1).reshape(-1))
        print(self.colors, '\n', self.traced_idxs)
        yield from self.traceEdges()
        while not self.traced_idxs.all():
            print('need to trace colors indexed by', 
                  np.argwhere(self.traced_idxs - 1).reshape(-1))
            print(self.colors, '\n', self.traced_idxs)
            u, v = self.chooseFromFractal()
            yield from self.traceCol(u)

    
# fill a frame with a pixelated fractal from the top
def fillFromTop(frame, pixelated_fractal, thickness=1):
    u_max, v_max = pixelated_fractal.aspect_ratio
    yield frame
    for v in range(0, v_max, thickness):
        for u in range(u_max):
            for dv in range(thickness):
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
            frame[u_t:u_t+w_t, v_t:v_t+h_t] \
                = resample(magnify, (w_t, h_t))
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
                                            n_colors=4,
                                            exponent=2, 
                                            julia_param=None,#0.285 + 0.01j,
                                            max_iter=40)
    #frame_iterator = fractalZoom(pixelated_fractal)
    frame = 255 * np.ones((960, 720, 3))
    
    # TODO: this is temporary
    
    print(pixelated_fractal.colors())
    
    def temp_frame_iterator():
        yield from fillFromTop(frame, pixelated_fractal, thickness=10)
        tracer = BorderTracer(frame, pixelated_fractal, speed=25)
        yield from tracer.generate()
        
    animate(aspect_ratio, temp_frame_iterator(), frame_rate=30)
    