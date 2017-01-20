import numpy as np
from utils import *
from features_extraction import *
from scipy.linalg.blas import sgemm

def get_loss_grad(noise, net, P, A, layer_content, style_weights, ratio, shape):
    """
    compute loss L = ratio*L_content + L_style and its grad with respect to noise with backpropagation
    Chain rule : L_style = sum_1^L w_l * El
                 L_content = 1./2 * (F^k-P)**2
                 d El / d X = d El / d Fl * d Fl / d X : see _get_grad_style for first term and back propagation for seconde one
                 d L_content / d X = d L_content / d Fk * d Fk / d X : same here
                 have to add El at layer l for each backward() to backpropagate on all Ek
    :param noise: image to optimize
    :param net: network
    :param P: content representation on content image
    :param A: style representation of style image
    :return: loss, grad
    """
    layers_style = style_weights.keys()
    layers = []
    for layer in reversed(net.blobs.keys()):
        if (layer in layers_style) or (layer == layer_content):
            layers.append(layer)

    noise = noise.reshape(shape)
    F, G = get_representation(net, noise, layers_style, layer_content)
    net.blobs[layers[0]].diff[0] = 0
    loss = 0
    grad = 0
    n_layer = 0
    for layer in layers:

        if layer is not layers[-1]:
            bottom_layer = layers[n_layer + 1]
        else:
            bottom_layer = None

        # copy the chunk to avoid trouble
        grad = net.blobs[layer].diff[0].copy()

        if layer in layers_style:
            # update L_style by computing El and d El / d Fl and adds it to net.blobs[layer].diff
            loss_, grad_ = get_loss_grad_style(A[layer], F[layer], G[layer])
            loss += style_weights[layer] * loss_
            grad += style_weights[layer] * grad_.reshape(grad.shape)

        if layer == layer_content:
            # update L_content by computing d L_content / d Fl and adds it to net.blobs[layer].diff
            loss_, grad_ = get_loss_grad_content(P[layer], F[layer])
            loss +=  ratio * loss_
            grad +=  ratio * grad_.reshape(grad.shape)

        net.blobs[layer].diff[0] = grad
        net.backward(start=layer, end=bottom_layer)
        n_layer += 1
        if bottom_layer is None:
            # over
            grad = net.blobs['data'].diff[0].copy()

    return loss, grad.flatten().astype(np.float64)

def get_loss_grad_content(P, F):
    """

    :param P: content representation of content image at layer of shape (Nl*Ml)
    :param F: content representation of generated image at layer of shape (Nl*Ml)
    :return: loss, grad of L_content(content,x,layer)
    """
    loss = 0.5 * np.sum((F-P)**2)
    grad = (F - P) * (F > 0)

    return loss, grad


def get_loss_grad_style(A, F, G):
    """
    :param A: style gram matrix of generated image at a certain layer l
    :param F: content representation of content image at layer l
    :param G: style gram matrix of generated image at layer l
    :return: loss, grad of L_style(style,x,layer)
    """
    cst = 1./ (4* (F.shape[0]**2) * (F.shape[1]**2))
    loss = cst * np.sum((G - A)**2)
    grad = 4 * cst * np.dot((G - A), F) * (F > 0)

    return loss, grad
