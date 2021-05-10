'''
Create animations out of sequences of numpy arrays.

TODO: dump animation to video file (with opencv?)
'''


import cv2


# TODO: allow up/downsampling to change window size?
def animate(
    aspect_ratio, 
    frame_iterator, 
    fps=10, 
    title='animation',
    video_dump=False):
    '''
    Display a series of 3D numpy arrays as RGB frames in a display window.
    
    Parameters
    ----------
    aspect_ratio : tuple
        (max_u, max_v)
    frame_iterator : iterator
        yields RGB frames contained in ndarrays of shape (max_u, max_v, 3)
    fps : int, optional
        Frame rate. The default is 10.
    title : str, optional
        Title of display window. The default is 'animation'.
    video_dump : bool, optional
        Whether to store frames as mp4 file. Default is False.
        If True, fps must be supported!
    '''
    out = cv2.VideoWriter(
            title + '.avi', 
            cv2.VideoWriter_fourcc(*'PIM1'), 
            fps, 
            aspect_ratio
            )
    for frame in frame_iterator:
        image = cv2.transpose(frame.astype('uint8'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if video_dump:
            out.write(image)
        cv2.imshow(title, image)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

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
    animate(
        aspect_ratio, 
        fuzzyFadeIn(aspect_ratio, p), 
        fps=32, 
        title='fuzzyFadeIn',
        video_dump=True
        )
            
        
        