from keras import Model
from keras.optimizers import RMSprop

from .image_encoder import ImageEncoder
from .sequence_decoder import SequenceDecoder

__all__ = [
    'NeuralImageCaptioning',
]


class NeuralImageCaptioning:
    """Neural Image Captioning
    
    The full model for generating code from sketch.
    
    Parameters:
    -----------
    embedding_dim : integer, the dimension in which to embed the sketch image and the tokens
    maxlen : integer, the maximum code length
    voc_size : integer, number of unique tokens in the vocabulary
    num_hidden_neurons : list with length of 2, specifying the number of hidden neurons in the LSTM decoders
    name : string, the name of the model, optional
    
    """

    def __init__(self, embedding_dim, maxlen, voc_size, num_hidden_neurons, word2idx, name='neural_image_captioning'):
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.voc_size = voc_size
        self.num_hidden_neurons = num_hidden_neurons
        self.word2idx = word2idx
        self.name = name

        # Encoder / Decoder
        self.image_encoder = ImageEncoder(embedding_dim).build_model()
        self.sequence_decoder = SequenceDecoder(maxlen, embedding_dim, voc_size, num_hidden_neurons, word2idx).build_model()

        # Inputs
        self.image_input = self.image_encoder.image_input
        self.sequence_input = self.sequence_decoder.sequence_input

        self.model = None

    def build_model(self):
        """Builds a Keras Model to train/predict"""

        image_embedding = self.image_encoder.model(self.image_input)
        sequence_output = self.sequence_decoder.model([self.sequence_input, image_embedding])

        self.model = Model([self.image_input, self.sequence_input], sequence_output, name=self.name)
        self.model.compile(RMSprop(1e-4), loss='categorical_crossentropy')
        return self
