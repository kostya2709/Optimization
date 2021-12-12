
from image import camns_image
from camns_lp import *
from image import *

if __name__ == "__main__":
    img1 = camns_image("cao1.jpg")
    img2 = camns_image("ksiwek1.jpg")
    #img1.show()
    img3 = camns_image("zhang1.jpg")
    imgs = mix_images([img1, img2, img3])
    imgs[0].show()
    #img.show()
    #img.write()