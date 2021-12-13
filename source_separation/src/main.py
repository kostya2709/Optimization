from image import camns_image, plot_images
from audio import camns_audio

def test_image():
    img1 = camns_image("cao1.jpg")
    img2 = camns_image("ksiwek1.jpg")
    img3 = camns_image("zhang1.jpg")
    plot_images([img1, img2, img3], 1, 3)
    
    mixed = camns_image.mix([img1, img2, img3])
    for i, img in enumerate(mixed):
        img.write("img_mixed_{}.jpg".format(i))
    unmixed = camns_image.camns_lp(mixed)
    
    plot_images([unmixed], 1, 3)
    for i, img in enumerate(unmixed):
        img.write("img_unmixed_{}.jpg".format(i))

def test_audio():
    rain = camns_audio("rain.wav")
    bob = camns_audio("bob.wav")
    city = camns_audio("city.wav")
    
    mixed = camns_audio.mix([rain, bob, city])
    for i, sound in enumerate(mixed):
        sound.write("mixed_{}.wav".format(i))
    
    unmixed = camns_audio.camns_lp(mixed)
    for sound, i in enumerate(unmixed):
        sound.write("unmixed_{}.wav".format(i))

if __name__ == "__main__":

    test_image()
    #test_audio()