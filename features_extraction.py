import numpy as np
from utils import *
from scipy.linalg.blas import sgemm


# def show_feature(net,transformer,layer,img):
#     """
#
#     :param net:
#     :param transformer:
#     :param layer:
#     :param img:
#     :return:
#     """
#
#     img = rescale_img(img, 800)
#     net, transformer = rescale_net(net, transformer, img)
#     content = transform(transformer, img)
#     f,_ = get_representation(net,content,[],layer)
#
#     print_image(img)

def get_representation(net, img, layers_style, content_layer):
    """
    Get feature maps of img from content_layer and compute its gram matrices from all the layers contained in
     layers_style
    :param net: loaded caffe network
    :param img: array (K*H*W) preprocessed image
    :param layers_style: list of str layers for style representation
    :param content_layer: str of layer used for content representation

    :return: F, G, dict of feature maps (F) and gram matrices (G) from all layers given in argument
    """
    net.blobs['data'].data[0] = img
    net.forward()
    G = {}
    F = {}
    for layer in layers_style:
        F[layer], G[layer] = gram(net, layer)

    if content_layer not in layers_style:
        F[content_layer] = feature_map(net, content_layer)

    return F, G

def gram(net,layer):
    """
    extract gram matrix from a certain layer and feature map
    :param net: network
    :param layer: str, layer from which to compute gram matrix

    :return: F, G feature map at layer and its gram matrix representation.
    """
    F = feature_map(net, layer)
    G = np.dot(F, F.T)

    return F, G


def feature_map(net,layer):
    """
    extract feature map at layer
    :param net: network
    :param layer: str, layer
    :return: F, feature map (Nl * Ml) where Nl is the size of the filter bank of the layer
    and Ml is the size of the ravelled filtered image.
    """
    F = net.blobs[layer].data[0].copy()
    F = F.reshape((F.shape[0], -1))

    return F
