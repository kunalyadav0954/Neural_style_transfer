# module for implementing helper functions for neural style transfer

import scipy.io as sio
import tensorflow as tf
import numpy as np
from PIL import Image


class CONFIG:
    image_height = 300
    image_width = 400
    color_channels = 3
    means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


def load_vgg_model(path):
    """
    Loads vgg model from the specified path and creates a graph by replacing max pool with average pool layers
    :param path: path to the pre-trained vgg-19 model, must be a .mat file
    :return: returns a dictionary that contains the graph for new model with max pool layers replaced with
             average pool layers

    The pre-trained vgg-19 model has following layers :

        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    """

    vgg = sio.loadmat(path)  # loads the vgg model into a dictionary
    vgg_layers = vgg['layers']  # getting layers key from the dictionary vgg

    # vgg_layers is an array that contains details about all layers
    # shape of vgg_layers - (1,43)

    def _weights(layer_number, expected_layer_name):
        """  Returns weights and bias from the vgg model for the specified layer """

        wb = vgg_layers[0][layer_number][0][0][2]  # array containing filters and bias
        w = wb[0][0]  # filter weights
        b = wb[0][1]  # bias weights

        layer_name = vgg_layers[0][layer_number][0][0][0][0]

        assert layer_name == expected_layer_name

        return w, b

    def _relu(conv_2d_layer):
        """ Returns a relu function wrapped over a conv 2d layer """

        return tf.nn.relu(conv_2d_layer)

    def _conv2d(prev_layer, layer_number, layer_name):
        """ Creates a conv2d layer over the previous layer specified by  prev_layer argument """
        w, b = _weights(layer_number, layer_name)
        w = tf.constant(w)
        b = tf.constant(np.reshape(b, (b.size)))

        return tf.nn.conv2d(prev_layer, filter=w, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer_number, layer_name):
        """ Returns conv2d + relu over the previous layer """

        return _relu(_conv2d(prev_layer, layer_number, layer_name))

    def _avgpool(prev_layer):
        """ average pool wrapper over previous layer """

        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructing the graph for our model to be used for neural style transfer
    graph = {}
    # input is made variable type so that it could be updated with subsequent iteration i.e trained
    graph['input'] = tf.Variable(np.zeros((1, CONFIG.image_height,
                                           CONFIG.image_width,
                                           CONFIG.color_channels)), dtype='float32')
    graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1'] = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])

    return graph


def load_image(path):
    """ loads the image from path and prepares it for our vgg model """
    img = Image.open(path)
    img = img.resize((CONFIG.image_width, CONFIG.image_height), Image.BICUBIC)
    img_data = np.array(img)
    raw = img_data  # without adding batch dimension and normalizing
    img_data = np.reshape(img_data, ((1,) + img_data.shape))  # adding batch dimension
    # subtracting mean to match the expected input of vgg
    img_data = img_data - CONFIG.means  # prepped image for our vgg model

    return raw, img_data


def generate_noise_image(content_input_to_vgg, noise_ratio=0.6):
    noise_image = np.random.uniform(-20, 20,
                                    (1, CONFIG.image_height, CONFIG.image_width, CONFIG.color_channels)).astype(
        'float32')

    input_image = (noise_image * noise_ratio) + (content_input_to_vgg * (1 - noise_ratio))

    return input_image
