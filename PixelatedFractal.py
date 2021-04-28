'''
Module for classes implementing the interface for pixelated fractals.
TODO: I might make a superclass for the interface with subclasses inheriting 
it if I want more different types of fractals?

TODO: arbitrary precision. Because 2^64 is only 0% of |C|.

Not that arbitrary precision would let us decide more than 0% of C...
'''

import numpy as np


# find min n for which |f^n(c)| > thres, max_iter if not found
def escapeTime(f, c, thres, max_iter):
    #if c == 0:   # avoid weird stuff
    #    return -1
    z = c
    for it in range(max_iter):
        z = f(z)
        if np.abs(z) > thres:
            return it
    return max_iter

class MandelbrotGreyscale:
    ''''
    Pixel-coloring object for visualizing multibrot and julia sets.
    
    Parameters
    ----------
    aspect_ratio: tuple
        Aspect ratio (u_max, v_max) to display.
    pixels_per_unit: numeric, optional
        Determines image scale. Default is u_max/4.
    n_colors: int, optional
        Determines number of lemniscates to draw in addition to the set itself.
        Default is 5.
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
                 n_colors=5,
                 window_center=0,
                 exponent=2, 
                 julia_param=None,
                 max_iter=100):
        self.aspect_ratio = aspect_ratio
        if pixels_per_unit is None:
            self.pixels_per_unit = aspect_ratio[0] / 4
        else:
            self.pixels_per_unit = pixels_per_unit
        self.n_colors = n_colors
        self.window_center = complex(window_center)
        self.exponent = exponent
        self.julia_param = julia_param
        self.max_iter = max_iter
    
    def adjustZoom(self, scale_factor, new_window_center):
        self.pixels_per_unit *= scale_factor
        self.window_center = complex(new_window_center)

    # convert pixel coords to point in the complex plane
    def pixelSpaceToC(self, u, v):
        u_max, v_max = self.aspect_ratio
        real_max = u_max/2 / self.pixels_per_unit + self.window_center.real
        imag_min = self.window_center.imag - v_max/2 / self.pixels_per_unit
        real = real_max - 1. * (u_max - u) / self.pixels_per_unit
        imag = imag_min + 1. * (v_max - v) / self.pixels_per_unit
        return real + imag*(0+1j)
    
    # get color of point from its escape time
    def colorFromTime(self, time):
        # code quality? never met her
        return 255 / self.n_colors * np.ceil(self.n_colors * (self.max_iter**0.5 - time**0.5) / self.max_iter**0.5)
    
    def pixelColor(self, u, v):
        point = self.pixelSpaceToC(u, v)
        if self.julia_param is None:
            f = lambda z : z**self.exponent + point
        else:
            f = lambda z : z**self.exponent + self.julia_param
        thres = 4
        time = escapeTime(f, point, thres, self.max_iter)
        return self.colorFromTime(time)
    
    # Return a list of all colors used. More external colors are later.
    def colors(self):
        result = np.array([self.colorFromTime(t) for t in range(self.max_iter + 1)])
        result = np.unique(result)
        return list(result)
    