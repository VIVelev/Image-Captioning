import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from .config import IMAGE_SIZE, TEXT_FILES_DIR
from .image import load_image

__all__ = [
    'load_raw_image_description_map',
    'init_image_descriptions_map',
    'init_word2idx',
    'init_idx2word',
    'load_set_images',
    'init_image_descriptions_set',
    'data_generator',
]


def load_raw_image_description_map():
    with open(TEXT_FILES_DIR + 'Flickr8k.token.txt', 'r') as flickr8_token:
        raw_image_description = flickr8_token.read().split('\n')[:-1]

    return raw_image_description

def init_image_descriptions_map():
    raw_image_description = load_raw_image_description_map()
    image_descriptions = dict()
    
    i = 0
    while i < len(raw_image_description):
        img_name = raw_image_description[i].split('.')[0]
        image_descriptions[img_name] = []
        
        while i < len(raw_image_description) and img_name == raw_image_description[i].split('.')[0]:
            descr = raw_image_description[i].split('\t')[1]
            image_descriptions[img_name].append(descr)
            i+=1
            
    return image_descriptions

def init_word2idx(vocabulary):
    return {val: key for key, val in enumerate(vocabulary)}
    
def init_idx2word(vocabulary):
    return {key: val for key, val in enumerate(vocabulary)}

def load_set_images(type):
    if type == 'train':
        filename = TEXT_FILES_DIR + 'Flickr_8k.trainImages.txt'
    elif type == 'dev':
        filename = TEXT_FILES_DIR + 'Flickr_8k.devImages.txt'
    else:
        filename = TEXT_FILES_DIR + 'Flickr_8k.testImages.txt'

    with open(filename, 'r') as f:
        img_names = f.read().split('\n')[:-1]
        
    img_names = [name.split('.')[0] for name in img_names]
    return img_names

def init_image_descriptions_set(set_images, image_descriptions):
    image_descriptions_set = dict()
    
    for img_name in set_images:
        image_descriptions_set[img_name] = []
        descriptions = image_descriptions[img_name]
        
        for descr in descriptions:
            image_descriptions_set[img_name].append(
                descr,
            )
    
    return image_descriptions_set

def data_generator(image_descriptions_set, word2idx, max_length, num_imgs_per_batch, voc_size):
    X_img = []
    X_seq = []
    Y_seq = []
    n = 0
    
    # loop for ever over images
    while True:
        for img_id, desc_list in image_descriptions_set.items():
            # retrieve the image
            img = load_image(img_id, preprocess=False)

            for desc in desc_list:
                X_img.append(img)
                
                # encode the sequence
                y_seq = [word2idx[word] for word in desc.split()] + [word2idx['<EOS>']]
                x_seq = [word2idx['<SOS>']] + y_seq[:-1]
                
                Y_seq.append(y_seq)
                X_seq.append(x_seq)
            
            n+=1
            if n == num_imgs_per_batch:
                X_seq = pad_sequences(X_seq, maxlen=max_length, padding='post')
                Y_seq = pad_sequences(Y_seq, maxlen=max_length, padding='post')

                # One-hot
                Y_seq = [[to_categorical(idx, voc_size) for idx in sent] for sent in Y_seq]

                yield [[np.array(X_img), X_seq], np.array(Y_seq)]

                X_img = []
                X_seq = []
                Y_seq = []
                n = 0
