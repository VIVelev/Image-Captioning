import pickle

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

from ..nn.inceptionv3_encoder import InceptionV3Encoder
from .config import IMAGE_SIZE, IMAGES_DIR

__all__ = [
    'load_image',
    'load_image_embedding_map',
]


def load_image(x, target_size=IMAGE_SIZE[:-1], preprocess=True):
    x = image.load_img(IMAGES_DIR+x+'.jpg', target_size=target_size)
    x = image.img_to_array(x)
    
    if preprocess:
        x = preprocess_input(x)

    return x

def load_image_embedding_map(set_type, image_descriptions_set):
    try:
        with open('./image_embedding_'+set_type+'.bin', 'rb') as f:
            image_embedding = pickle.load(f)
        print('"{}" Image-Embedding Map loaded.'.format(set_type))        

    except FileNotFoundError:
        print('Creating "{}" Image-Embedding Map...'.format(set_type))

        encoder = InceptionV3Encoder()
        image_embedding = dict()
        for img_id, _ in image_descriptions_set.items():
            img = load_image(img_id, preprocess=True)
            image_embedding[img_id] = encoder.encode_image(img)

        with open('./image_embedding_'+set_type+'.bin', 'wb') as f:
            pickle.dump(image_embedding, f)

        print('Done.')
    
    return image_embedding
