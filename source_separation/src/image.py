import cv2
import numpy as np
import pathlib
from camns_lp import *
import matplotlib.pyplot as plt

def plot_images(imgs, rows, columns):
    # create figure
    fig = plt.figure(figsize=(10, 7))

    for i in range(len(imgs)):
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, i + 1)
        
        # showing image
        plt.imshow(imgs[i].image_matrix__)
        plt.axis('off')


class camns_image(camns_object):
    SAVED = 0

    @staticmethod
    def tovector(img_matrix, channel_num=1):
        PIXEL_LEN = 3
        img_size = img_matrix.shape[0] * img_matrix.shape[1]
        img = img_matrix.reshape((img_size, PIXEL_LEN))
        result = np.zeros(img.shape[0] * channel_num)
        last = 0
        for pixel in img:
            for component in range(channel_num):
                result[last] = pixel[component]
                last += 1
        return result

    @staticmethod
    def tomatrix(img_vector, channel_num=1, size=None):
        PIXEL_LEN = 3
        if size is None:
            width = int((img_vector.shape[0] / channel_num) ** (1 / 2))
            size = (width, width)

        pixel_array = img_vector.reshape(-1, channel_num)
        if channel_num < PIXEL_LEN:
            res = np.empty((pixel_array.shape[0], 3))
            for pixel in range(pixel_array.shape[0]):
                tmp = list(pixel_array[pixel])
                for _ in range(3 - channel_num):
                    tmp.append(pixel_array[pixel][0])
                res[pixel] = np.array(tmp)
        else:
            res = pixel_array

        res = res.reshape(size[0], size[1], PIXEL_LEN)
        return res

    @staticmethod
    def to_imgs_matrix(imgs):
        """
        imgs = [img1, img2, ...]
        output: (L, M) - L - img size, M - number of images
        """
        return np.array(list(map(lambda x: x.vector__, imgs))).T

    @staticmethod
    def from_imgs_matrix(imgs, channel_num = 1):
        """
        input: (L, M) - L - img size, M - number of images
        imgs = [img1, img2, ...]
        """
        return np.array(list(map(lambda x: camns_image(channel_num=channel_num).set_pixels(x), imgs.T)))

    @staticmethod
    def mix(imgs, observ_num=None):
        pixels = camns_image.to_imgs_matrix(imgs)
        res = get_random_observations(pixels, observ_num)
        mixed = camns_image.from_imgs_matrix(res, imgs[0].channel_num__)
        return mixed

    @staticmethod
    def camns_lp(imgs, observ_num=None):
        pixels = camns_image.to_imgs_matrix(imgs)
        res = camns_lp(pixels, observ_num)
        unmixed = camns_image.from_imgs_matrix(res)
        return unmixed

    def __init__(self, file: str=None, folder=None, channel_num=1):
        super().__init__()
        self.channel_num__ = channel_num

        if file is not None:

            if folder is None:
                file = pathlib.Path(__file__).parent.parent.resolve() / 'images' / file
            else:
                file = pathlib.PurePath(folder, file)

            self.image_matrix__ = cv2.imread(str(file), cv2.IMREAD_COLOR)
            self.vector__ = camns_image.tovector(self.image_matrix__, self.channel_num__) / 255
            self.size__ = self.vector__[0]
            self.has_image__ = True
        else:
            self.has_image__ = False

    def set_pixels(self, pixels=np.ndarray, size=None):

        self.image_matrix__ = camns_image.tomatrix(pixels, self.channel_num__, size) * 255
        self.vector__ = pixels
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
        
        cv2.waitKey(0)

    def write(self, name=None):
        if name is None:
            name = 'res{}.jpg'.format(camns_image.SAVED)
            camns_image.SAVED += 1
        file = pathlib.Path(__file__).parent.parent.resolve() / 'res' / name
        cv2.imwrite(str(file), self.image_matrix__)
        
    def __del__(self):
        """ It is for removing/deleting created GUI window from screen and memory
        """
        cv2.destroyAllWindows()