from skimage.transform import rescale,resize
from utils import *
from features_extraction import *
from loss import *
from collections import OrderedDict
from scipy.optimize import minimize
from time import time
import datetime
import progressbar as pb


def get_bounds(mean, shape):
    """
    Get the bounds of the minimization problem. Usually each pixel pixel is within a range [0 255]. But in the network
    we remove the mean of the images in the ImageNet data set so that the range becomes [-mean 255-mean].

    :param mean: (K*H*W) array of mean of the data set used to train the network.
    :param shape: (K,H,W) tuple of the preprocessed generated image

    :return bounds: (K*H*W,) list of bounds for each pixels of the ravelled generated image
    """
    # assume img is BGR (K*H*W) with pixel's range 255
    min_B = -mean[0]
    min_G = -mean[1]
    min_R = -mean[2]

    max_B = min_B + 255
    max_G = min_G + 255
    max_R = min_R + 255

    bounds_B = (min_B,max_B)
    bounds_G = (min_G,max_G)
    bounds_R = (min_R,max_R)

    idx_B = shape[1]*shape[2]
    idx_G = 2*shape[1]*shape[2]
    idx_R = 3*shape[1]*shape[2]

    bounds = []
    for k in range(np.prod(shape)):
        if k<idx_B:
            bounds.append(bounds_B)
        elif k<idx_G:
            bounds.append(bounds_G)
        else:
            bounds.append(bounds_R)

    return bounds


def create_noise(t):
    """
    Create a noise. Each pixel are chosen independently with normal distribution
    :param t : tuple of net image dimension
    :return noise: white noise of dim t
    """

    mean = 0
    std = 1
    n = np.prod(t)
    samples = np.sqrt(abs(np.random.normal(mean, std, n)**2))
    noise = samples.reshape(t)
    noise[:, :, 0] = noise[:, :, 0]/max(noise[:, :, 0].ravel())
    noise[:, :, 1] = noise[:, :, 1]/max(noise[:, :, 1].ravel())
    noise[:, :, 2] = noise[:, :, 2]/max(noise[:, :, 2].ravel())

    return noise

def progressbar_update(f,progressbar):
    f['it'] += 1
    try:
        progressbar.update(f['it'])
    except:
        progressbar.finished = True


def transfer(content,
             style,
             net,
             transformer,
             layer_content,
             style_weights,
             start,
             ratio,
             lengths,
             style_scale,
             optimization
             ):
    """

    :param content: content image (H*W*K)
    :param style: style image (H'*W"*K')
    :param net: pretrained loaded network (caffe framework)
    :param transformer: transformer of the loaded network model
    :param layer_content: str, which layer to represent the content
    :param style_weights: OrderedDict of ('layer',float) indicating which layer to use for style representation
     and which weight we apply to it
    :param start: str or array (H*W*K), str must be 'mixed' to mix content and style for the initialization of the
    transfer, 'content' to start from the content image, or 'random' to start from a white noise.
    :param ratio: float, high ratio will result in generating content image and low ratio will result in generating
    style image's texture.
    :param lengths: float, limit of number of pixels along the two sides of the generated image.
    :param style_scale: float, scale of style image compared to content image
    :param optimization: dict of scipy.optimize.minimize parameters.

    :return: generated image of size (H*W*K)
    """
    net = clear(net)
    t = content.shape
    layers_style = style_weights.keys()

    if start is 'random':

        noise = create_noise(t)

    elif start is 'content':

        noise = content.copy()

    elif start is 'mixed':
        style_ = resize(style,content.shape[0:2])
        noise = 0.95 * content.copy() + 0.5 * style_.copy()

    else:
        noise = start

    # compute content features
    content = rescale_img(content, lengths)
    net, transformer = rescale_net(net, transformer, content)
    content = transform(transformer, content)
    P, _ = get_representation(net, content, layers_style, layer_content)

    # compute style features
    scale = int(lengths * style_scale)
    print style.shape
    style = rescale_img(style, scale)
    print style.shape
    net, transformer = rescale_net(net, transformer, style)
    style = transform(transformer, style)
    _, A = get_representation(net, style, layers_style, layer_content)

    # prepare noise and rescale net
    noise = rescale_img(noise, lengths)
    net, transformer = rescale_net(net, transformer, noise)
    noise = transform(transformer, noise)
    noise_shape = noise.shape

    # set arguments of get_loss_grad
    args = (net,
            P,
            A,
            layer_content,
            style_weights,
            ratio,
            noise_shape
            )

    # we need to set constraints otherwise the gradient takes wrong direction
    bounds = get_bounds(transformer.mean['data'].ravel(), noise.shape)

    # each step is updating net.blobs['data] with the updated x
    progressbar = pb.ProgressBar()
    progressbar.widgets = ["Optimizing: ", pb.Percentage(),
                            " ", pb.Bar(marker=pb.AnimatedMarker()),
                            " ", pb.ETA()]
    progressbar.maxval = optimization['maxiter']

    f = {'it':0}
    callback = lambda x : progressbar_update(f,progressbar)

    progressbar.start()
    out = minimize(fun=get_loss_grad,
                   x0=noise.ravel(),
                   method='L-BFGS-B',
                   jac=True,
                   bounds=bounds,
                   options=optimization,
                   args=args,
                   callback=callback
                   )
    progressbar.finish()
    # generate image
    output = generate_img(transformer, net.blobs['data'].data[0])

    return output
