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
    
    Image Encoder Model - An InceptionV3 CNN pretrained on ImageNet.
    
    Parameters:
    -----------
    embedding_dim : integer, the dimension in which to embed the image and the tokens
    n_top_trainable_layers : integer, number of the top layers which weights will be updated during training (fine-tuned)
    name : string, the name of the model, optional
    
    """

    def __init__(self, embedding_dim, n_top_trainable_layers=16, name='image_encoder'):
        self.embedding_dim = embedding_dim
        self.n_top_trainable_layers = n_top_trainable_layers
        self.name = name
        
        # Inputs
        self.image_input = Input(IMAGE_SIZE, name='image_input')

        # Get the InceptionV3 model trained on imagenet data
        self.inceptionv3 = InceptionV3(weights='imagenet', include_top=True, input_tensor=self.image_input)
        # Cut till 2048 Dense Layer
        self.inceptionv3 = Model(self.inceptionv3.input, self.inceptionv3.layers[-2].output)

        # Top
        self.dropout_encoder = Dropout(0.5, name='dropout_encoder')
        self.dense_encoder = Dense(self.embedding_dim, name='dense_encoder')
        self.relu_encoder = Activation('relu', name='relu_encoder')
        self.reshape = Reshape((1, self.embedding_dim), name='reshape')

        self.model = None

    def build_model(self):
        x = self.inceptionv3.output
        x = self.dropout_encoder(x)
        x = self.dense_encoder(x)
        x = self.relu_encoder(x)
        x = self.reshape(x)

        self.model = Model(self.image_input, x, name=self.name)

        # Fix
        for layer in self.model.layers[:-self.n_top_trainable_layers]:
            layer.trainable = False
        # Fine-tune
        for layer in self.model.layers[-self.n_top_trainable_layers:]:
            layer.trainable = True

        self.model.compile(RMSprop(1e-4), loss='categorical_crossentropy')
        return self
