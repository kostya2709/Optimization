import cv2
import numpy as np
import pathlib
from camns_lp import *


def norm(x):
    if x > 1:
        return 1
    if x < 0:
        return 0
    return x

class pixel():
    def __init__(self, colour):
        self.R = colour[0]
        self.G = colour[1]
        self.B = colour[2]
    

    def __add__(self, other):
        return pixel([norm(self.R + other.R), norm(self.G + other.G), norm(self.B + other.B)])

    def __sub__(self, other):
        return pixel([norm(self.R - other.R), norm(self.G - other.G), norm(self.B - other.B)])
    
    def __mul__(self, const):
        return pixel([norm(self.R * const), norm(self.G * const), norm(self.B * const)])

    def __truediv__(self, const):
        return pixel([norm(self.R / const), norm(self.G / const), norm(self.B / const)])

    def __str__(self):
        return "[{} {} {}]".format(self.R, self.G, self.B)
    
    def __repr__(self):
        return "[{} {} {}]".format(self.R, self.G, self.B)

    def tolist(self):
        return [self.R, self.G, self.B]



def mix_images(imgs, observ_num=None):
    print(imgs[0].image_matrix__.shape)
    pixels = np.array(list(map(lambda x: x.vector__, imgs)))
    pixels = np.array(pixels).T
    res = get_random_observations(pixels, observ_num)
    mixed = list(map(lambda x: camns_image().set_pixels(x), res.T))
    return mixed



class camns_image(camns_object):
    def __init__(self, file: str=None, folder=None):
        super().__init__()

        if file is not None:

            if folder is None:
                file = pathlib.Path(__file__).parent.parent.resolve() / 'images' / file
            else:
                file = pathlib.PurePath(folder, file)

            self.image_matrix__ = cv2.imread(str(file), cv2.IMREAD_COLOR)
            self.height__ = self.image_matrix__.shape[0]
            self.width__ = self.image_matrix__.shape[1]
            self.pixel_width__ = self.image_matrix__.shape[2]
            self.size__ = self.height__ * self.width__
            self.image_vector__ = self.image_matrix__.reshape((self.size__, self.pixel_width__))
            self.vector__ = self.image_vector__ / 255
            self.vector__ = np.array(list(map(lambda x: pixel([x[0], x[1], x[2]]), self.vector__)))
            self.has_image__ = True
        else:
            self.has_image__ = False


    def set_pixels(self, pixels=np.ndarray, size=None):
        self.size__ = pixels.shape[0]

        if size is None:
            width = self.size__ ** (1 / 2)
            size = (width, width)

        self.height__ = int(size[0])
        self.width__ = int(size[1])
        self.pixel_width__ = 3
        self.vector__ = pixels
        self.image_vector__ = np.array(list(map(lambda x: x.tolist(), pixels)))
        self.image_matrix__ = (self.image_vector__ * 255).astype(np.uint8).reshape(self.width__, self.height__, self.pixel_width__,)
        self.has_image__ = True

        return self

    def show(self, title=""):
        """
        Creating GUI window to display an image on screen
        First Parameter is windows title (should be in string format)
        Second Parameter is image array
        """
        assert self.has_image__ == True, "You cannot show an image, which was not set before!"
        cv2.imshow(title, self.image_matrix__)
        
        # To hold the window on screen, we use cv2.waitKey method
        # Once it detected the close input, it will release the control
        # To the next line
        # First Parameter is for holding screen for specified milliseconds
        # It should be positive integer. If 0 pass an parameter, then it will
        # hold the screen until user close it.
        cv2.waitKey(0)

    def write(self):
        cv2.imwrite('grayscale.jpg', self.image_matrix__)
        
    def __del__(self):
        """ It is for removing/deleting created GUI window from screen and memory
        """
        cv2.destroyAllWindows()