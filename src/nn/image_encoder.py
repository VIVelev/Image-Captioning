from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Activation, Dense, Dropout, Input, Reshape
from keras.optimizers import RMSprop

from ..utils.config import IMAGE_SIZE

__all__ = [
    'ImageEncoder',
]


class ImageEncoder:
    """Image Encoder
    
    Image Encoder Model.
    
    Parameters:
    -----------
    embedding_dim : integer, the dimension in which to embed the image image and the tokens
    name : string, the name of the model, optional
    
    """

    def __init__(self, embedding_dim, name='image_encoder'):
        self.embedding_dim = embedding_dim
        self.name = name

        self.image_input = Input(IMAGE_SIZE, name='image_input')

        # Get the InceptionV3 model trained on imagenet data
        self.inceptionv3 = InceptionV3(weights='imagenet')
        # Remove the last layer (output softmax layer) from the inception v3
        self.inceptionv3 = Model(self.inceptionv3.input, self.inceptionv3.layers[-2].output, name='inceptionv3')

        self.dropout_encoder = Dropout(0.5, name='dropout_encoder')
        self.dense_encoder = Dense(self.embedding_dim, name='dense_encoder')
        self.relu_encoder = Activation('relu', name='relu_encoder')
        self.reshape = Reshape((1, self.embedding_dim), name='reshape')

        self.model = None

    def build_model(self):
        x = self.inceptionv3(self.image_input)
        x = self.dropout_encoder(x)
        x = self.dense_encoder(x)
        x = self.relu_encoder(x)
        x = self.reshape(x)

        self.model = Model(self.image_input, x, name=self.name)
        self.model.compile(RMSprop(1e-4), loss='categorical_crossentropy')
        return self
