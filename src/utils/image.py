from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

from .config import IMAGE_SIZE, IMAGES_DIR

__all__ = [
    'load_image',
]


def load_image(x, target_size=IMAGE_SIZE[:-1], preprocess=True):
    x = image.load_img(IMAGES_DIR+x+'.jpg', target_size=target_size)
    x = image.img_to_array(x)
    
    if preprocess:
        x = preprocess_input(x)

    return x
