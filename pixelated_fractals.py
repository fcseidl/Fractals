'''
Module for classes implementing the interface for pixelated fractals.

TODO: I might make a superclass for the interface with subclasses inheriting 
it if I want more different types of fractals?

TODO: cycle detection for quick termination of escape time algorithm
TODO: arbitrary precision.
TODO: option for coloring based on how far away we get in fixed time

TODO: make it faster! Exploit numpy broadcasting as much as possible
'''

import numpy as np
from numpy import random


class PixelatedFractal:
    ''''
    Pixel-coloring object for visualizing multibrot and multi-Julia sets.
    
    Parameters
    ----------
    aspect_ratio: tuple
        Aspect ratio (u_max, v_max) to display.
    color_cycle: list, optional
        A point which diverges in k iterations will be given the color 
        specified by color_cycle[k % len(color_cycle)]. By default the only 
        color used will be white.
    pixels_per_unit: numeric, optional
        Determines image scale. Default is u_max/4.
    window_center: complex
        Display centers on this point. Default is 0.
    exponent: complex, optional, default is 2.
    julia_param: complex, optional, default is None.
        Pixels are colored by the escape time of z_n+1 = z_n**exponent + c,
        where z_0 is the point specified by (u,v) and c is z_0 by default 
        (multibrot set) or julia_param if it exists (julia set).
    max_iter: int
        Maximum number of iterations to use in escape timing.
        
    Methods
    -------
    adjustZoom(scale_factor, new_window_center)
        zoom in by a scale factor towards a new center
    pixelSpaceToC(u, v)
        convert pixel coordinates to the complex plane
    pixelColor(u, v)
        return the color of a single pixel using escape time algorithm
    
    TODO: update docstring!
    '''
    
    NOT_ESCAPED = np.infty
    
    def __init__(self, 
                 aspect_ratio, 
                 color_cycle=[np.array([255,255,255])],
                 pixels_per_unit=None, 
                 window_center=0,
                 exponent=2, 
                 julia_param=None,
                 max_iter=100):
        # describe fractal as a subset of C
        self.exponent = exponent
        self.julia_param = julia_param
        self.max_iter = max_iter
        # describe visualization in pixel space
        self.aspect_ratio = aspect_ratio
        if pixels_per_unit is None:
            self.pixels_per_unit = aspect_ratio[0] / 4
        else:
            self.pixels_per_unit = pixels_per_unit
        self.color_cycle = color_cycle
        self.adjustZoom(1, window_center)
        
    # create a deep copy of this object
    def deepcopy(self):
        return PixelatedFractal(
            self.aspect_ratio,
            self.color_cycle,
            self.pixels_per_unit,
            self.window_center,
            self.exponent,
            self.julia_param,
            self.max_iter,
        )
    
    # Change resolution by a factor of sf without moving viewport
    def adjustResolution(self, sf):
        self.aspect_ratio = (int(self.aspect_ratio[0] * sf),
                             int(self.aspect_ratio[1] * sf))
        self.pixels_per_unit *= sf
    
    # adjust zoom to new center and scale factor
    def adjustZoom(self, scale_factor, new_window_center):
        self.pixels_per_unit *= scale_factor
        self.window_center = complex(new_window_center)
        
    # choose a random point in frame from a fractal by rejection sampling
    def chooseFromFractal(self):
        u_max, v_max = self.aspect_ratio
        u, v = (0, 0)
        while (self.pixelColor(u, v) != 0).any():
            u = random.randint(u_max)
            v = random.randint(v_max)
        return u, v
    
    # convert pixel coords to point in the complex plane
    def pixelSpaceToC(self, u, v):
        u_max, v_max = self.aspect_ratio
        real_max = u_max/2 / self.pixels_per_unit + self.window_center.real
        imag_min = self.window_center.imag - v_max/2 / self.pixels_per_unit
        real = real_max - 1. * (u_max - u) / self.pixels_per_unit
        imag = imag_min + 1. * (v_max - v) / self.pixels_per_unit
        return real + imag*(0+1j)
    
    # reverse coord transform
    def cToPixelSpace(self, z):
        u_max, v_max = self.aspect_ratio
        z = np.atleast_1d(z)
        real, imag = z.real, z.imag
        real_max = u_max/2 / self.pixels_per_unit + self.window_center.real
        imag_min = self.window_center.imag - v_max/2 / self.pixels_per_unit
        u = (real - real_max) * self.pixels_per_unit + u_max
        v = v_max - (imag - imag_min) * self.pixels_per_unit
        u = u.astype(int)
        v = v.astype(int)
        if u.shape[0] == 1:
            u, v = u[0], v[0]
        return u, v
    
    # map to iterate at a point
    def get_map(self, point):
        if self.julia_param is None:
            return lambda z : z**self.exponent + point
        else:
            return lambda z : z**self.exponent + self.julia_param
    
    # return list of tuples representing pixels along orbit
    def orbit_pixels(self, u, v, length):
        point = self.pixelSpaceToC(u, v)
        f = self.get_map(point)
        orbit = [point]
        while len(orbit) < length:
            point = f(point)
            orbit.append(point)
        u, v = self.cToPixelSpace(np.array(orbit))
        return list(zip(u, v))
    
    # find min n for which |f^n(c)| > 2, or NOT_ESCAPED sentinel value
    def escapeTime(self, point):
        f = self.get_map(point)
        z = point
        for it in range(self.max_iter):
            z = f(z)
            if np.abs(z) > 2:
                return it
        return self.NOT_ESCAPED
    
    # get color of point from its escape time
    def colorFromTime(self, time):
        if time == self.NOT_ESCAPED:
            return np.zeros(3)
        return self.color_cycle[time % len(self.color_cycle)]
        
    # return color of a pixel
    def pixelColor(self, u, v):
        point = self.pixelSpaceToC(u, v)
        time = self.escapeTime(point)
        return self.colorFromTime(time)
