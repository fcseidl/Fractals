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
        times = sorted([self.escapeTime(p) for p in points])
        # set quantiles for histogram coloring
        self.time_quantiles = []
        for n in range(1, self.n_colors + 1):
            quantile = times[n * int(n_samp / (self.n_colors + 1))]
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
    
    # find min n for which |f^n(c)| > 4, max_iter if not found
    def escapeTime(self, point):
        if self.julia_param is None:
            f = lambda z : z**self.exponent + point
        else:
            f = lambda z : z**self.exponent + self.julia_param
        z = point
        for it in range(self.max_iter):
            z = f(z)
            if np.abs(z) > 2:
                return it
        return np.infty    # TODO: this is a bit of a hack
    
    # get color of point from its escape time
    def colorFromTime(self, time):
        #return 255 / self.n_colors * np.ceil(self.n_colors * (self.max_iter - time) / self.max_iter)
        color = np.array([0, 255, 0])
        dimness = (self.time_quantiles > time).sum() / (self.n_colors + 1)
        return color * dimness
        
    # return color of a pixel
    def pixelColor(self, u, v):
        point = self.pixelSpaceToC(u, v)
        time = self.escapeTime(point)
        return self.colorFromTime(time)
    
    # Return a list of all colors used. More external colors are earlier.
    def colors(self):
        result = [self.colorFromTime(t) for t in range(self.max_iter)]
        result.append(np.zeros(3))
        result = np.unique(result, axis=0)
        return list(result)