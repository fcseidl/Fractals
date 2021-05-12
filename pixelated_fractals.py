'''
Module for classes implementing the interface for pixelated fractals.

TODO: I might make a superclass for the interface with subclasses inheriting 
it if I want more different types of fractals?

TODO: cycle detection for quick termination of escape time algorithm
TODO: arbitrary precision. Because 2^64 is only 0% of |C|. Not that arbitrary 
precision would let us decide more than 0% of C...
TODO: option for coloring based on how far away we get in fixed time
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
    pixels_per_unit: numeric, optional
        Determines image scale. Default is u_max/4.
    n_colors: int, optional
        Determines number of lemniscates to draw in addition to the set itself.
        Default is 7.
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
    color(u, v)
        return the color of a pixel using escape time algorithm
    '''
    
    ESCAPED = np.infty
    
    def __init__(self, 
                 aspect_ratio, 
                 pixels_per_unit=None, 
                 n_colors=7,
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
        self.n_colors = n_colors
        self.adjustZoom(1, window_center)
    
    # adjust zoom and recalculate quantiles for histogram coloring
    # TODO: only consider external points in the histogram!
    def adjustZoom(self, scale_factor, new_window_center):
        # set view window
        self.pixels_per_unit *= scale_factor
        self.window_center = complex(new_window_center)
        # randomly sample a bunch of escape times
        u_max, v_max = self.aspect_ratio
        n_samp = 15 * self.max_iter * self.n_colors
        u = random.randint(u_max, size=n_samp)
        v = random.randint(v_max, size=n_samp)
        points = self.pixelSpaceToC(u, v)
        times = np.array([self.escapeTime(p) for p in points])
        times = times[times != self.ESCAPED]
        # set quantiles for histogram coloring
        self.time_quantiles = []
        for n in range(1, self.n_colors):
            quantile = times[n * int(len(times) / self.n_colors)]
            if n > 1:
                floor = self.time_quantiles[-1] + 1
            else:
                floor = 0
            self.time_quantiles.append(max(quantile, floor))
        self.time_quantiles.append(self.max_iter + self.n_colors)
        self.time_quantiles = np.array(self.time_quantiles)
        
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
    
    # return  list of tuples representing pixels along orbit
    def orbit_pixels(self, u, v, length):
        point = self.pixelSpaceToC(u, v)
        f = self.get_map(point)
        orbit = [point]
        while len(orbit) < length:
            point = f(point)
            orbit.append(point)
        u, v = self.cToPixelSpace(np.array(orbit))
        return list(zip(u, v))
    
    # find min n for which |f^n(c)| > 2, max_iter if not found
    def escapeTime(self, point):
        f = self.get_map(point)
        z = point
        for it in range(self.max_iter):
            z = f(z)
            if np.abs(z) > 2:
                return it
        return self.ESCAPED
    
    # get color of point from its escape time
    # TODO: avoid harcoded constant colors!
    def colorFromTime(self, time):
        if time == self.ESCAPED:
            return np.zeros(3)
        interior = np.array([255, 0, 0])
        exterior = np.array([0, 0, 255])
        mix = (self.time_quantiles > time).sum() / self.n_colors
        return interior * (1 - mix) + exterior * mix    # linear interpolation
        
    # return color of a pixel
    def pixelColor(self, u, v):
        point = self.pixelSpaceToC(u, v)
        time = self.escapeTime(point)
        return self.colorFromTime(time)
    
    # Return a list of all colors used. More external colors are earlier.
    def colors(self):
        result = [self.colorFromTime(t) for t in range(self.max_iter)]
        result.append(self.colorFromTime(self.ESCAPED))
        result = np.unique(result, axis=0)
        return list(result)
