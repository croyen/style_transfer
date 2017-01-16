import matplotlib.pyplot as plt
import numpy as np
import caffe
import os
import subprocess
from skimage.io import imsave
from skimage.transform import rescale,resize
from sympy.ntheory import factorint
from google.protobuf import text_format
from collections import OrderedDict


def build_net():
    """

    :return: caffe VGG19 pretrained network without fc layers and transformer for processing image for the network.
    """
    model_path = os.getcwd() + '/models/vgg19/'
    proto_model = model_path +'VGG_ILSVRC_19_layers_deploy.prototxt'
    mean_model = model_path +'ilsvrc_2012_mean.npy'
    pretrained_model = model_path +'VGG_ILSVRC_19_layers.caffemodel'

    net = caffe.Net(proto_model,pretrained_model,caffe.TEST)
    print('vgg19 model successfully loaded \n')

    # Transformer for data
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2,0,1)) # transform H*W*K imported image to K*H*W for network
    transformer.set_channel_swap('data',(2,1,0)) #transform RGB to BGR

    ms = np.load(mean_model)
    transformer.set_mean('data',np.array([np.mean(ms[0]),np.mean(ms[1]),np.mean(ms[2])])) #remove mean of training data
    transformer.set_raw_scale('data',255) #scaling data to values in [0 256]

    return net, transformer


def net_show():
    """
    plot and return the vgg19 architecture.
    :return: image ((H*W*K) array) of the net archi
    """
    model_base_path = '/home/kroy/MVA/Recvis/Projet/models/vgg19'
    proto_model = os.path.join(model_base_path,'VGG_ILSVRC_19_layers_deploy.prototxt')
    mean_model = os.path.join(model_base_path,'ilsvrc_2012_mean.npy')
    pretrained_model = os.path.join(model_base_path,'VGG_ILSVRC_19_layers.caffemodel')

    return_code = subprocess.call(["python /home/kroy/apps/caffe/python/draw_net.py",proto_model,"my_net.py"])
    net_img = caffe.io.imread("my_net.png")
    plt.figure()
    plt.imshow(net_img)
    plt.axis('off')
    plt.show()
    return net_img


def print_image(imgs,subformat=None):
    """
    print image or list of images.

    :param imgs: array or list of array of type (H*W*K) to show()
    :param subformat: tuple (p,m), format of the subplot ( plt.subplot(p,m,k) for k=1:len(imgs). If None (default) then
    p and m are automatically computed taking m as the biggest factor of len(imgs).

    """
    if type(imgs) is list:
        if subformat is None:
            n = len(imgs)
            f = factorint(n)
            m = max(f.keys())
            m = m**(f[m])
            p = n/m
        else:
            if np.prod(subformat) != len(imgs):
                print 'Error, subformat does not respect number of images to print'
                pass
            n = len(imgs)
            p = subformat[0]
            m = subformat[1]


        plt.figure(1, figsize=(12, 12))
        for k in range(n):
            plt.subplot(p, m, k+1)
            plt.imshow(imgs[k])
    else:
        plt.figure(1,figsize=(12, 12))
        plt.imshow(imgs)
    plt.show()

def rescale_img(img, max_length = 512.):
    """
    rescale img so that its sides does not exceed a certain length. Preserve proportion.

    :param img: array of content image (H*W*K) (default format)
    :param max_length: float, limit for H and W.

    :return: rescale img
    """
    max_length = float(max_length)
    max_dim = float(max(img.shape[:2]))

    img = rescale(img, max_length / max_dim)

    return img


def transform(transformer, img):
    """

    :param transformer:
    :param img: Default (H*W*K) RGB image
    :return: preprocessed image ((K*H*W), centered, BGR, [0 255] pixel range image)
    """
    return transformer.preprocess('data', img)


def backtransform(transformer,img):
    """
    Inverse transformation
    :param transformer:
    :param img:
    :return: deprocessed (H*W*K) RGB image
    """
    return transformer.deprocess('data', img)


def generate_img(transformer, output):

    output = backtransform(transformer, output)
    print('output image of shape: '+str(output.shape))
    return output


def rescale_net(net, transformer, input_img):
    """
    fit network and transformer to input image
    :param net:
    :param transformer:
    :param input_img: input of type blob['data'] (processed image)
    :return: net, transformer
    """
    dims = (1, input_img.shape[2], input_img.shape[0], input_img.shape[1])
    net.blobs['data'].reshape(*dims)
    transformer.inputs['data'] = net.blobs['data'].data.shape

    return net, transformer

def clear(net):
    """
    Clear all blobs (data and diff)
    :param net:
    :return: net
    """
    for layer in net.blobs:
        net.blobs[layer].data[0] = 0
        net.blobs[layer].diff[0] = 0

    return net

def load(name,return_name=False):
    """
    load an image
    :param name: str of the image contained in ./images/
    :param return_name: whether you want to return the name of not
    :return: image and its name if wanted (used for args function)
    """
    path = 'images/'+name+'.jpg'
    img = caffe.io.load_image(path)
    if return_name is True:
        return img, name
    else:
        return img

class args():
    """
    Define an object to store all the parameters of the transfer so that you can access and change them more easily.
    """
    def __init__(self,net,transformer):
        self.content, self.content_name = load('tubingen',True)
        self.style, self.style_name = load('starry_night',True)
        self.net = net
        self.transformer = transformer
        # parameters of transfer
        self.start = 'random'
        self.ratio = 1e-3
        self.lengths = 512.
        self.style_scale = 1.
        self.content_layer='conv4_2'
        self.style_weights=OrderedDict([('conv1_1', 0.2),
                                   ('conv2_1', 0.2),
                                   ('conv3_1', 0.2),
                                   ('conv4_1', 0.2),
                                   ('conv5_1', 0.2)
                                   ])


        #parameters of optimization
        self.optimization = {'disp': False,
                        'maxls': 20,
                        'iprint': -1,
                        'gtol': 1e-05,
                        'eps': 1e-08,
                        'maxiter': 500,
                        'ftol': 2.220446049250313e-09,
                        'maxcor': 8,
                        'maxfun': 15000}
    def change_content(self,name):
        self.content, self.content_name = load(name,True)

    def change_style(self,name):
        self.style, self.style_name = load(name,True)

    def get(self):
        """return a tuple of transfer ordered parameters"""
        # build OrderedDict for convenience
        params = OrderedDict([('content', self.content),
                              ('style',self.style),
                              ('net',self.net),
                              ('transformer',self.transformer),
                              ('content_layer',self.content_layer),
                              ('style_weights',self.style_weights),
                              ('start',self.start),
                              ('ratio',self.ratio),
                              ('lengths',self.lengths),
                              ('style_scale',self.style_scale),
                              ('optimization',self.optimization)])
        return tuple(params.values())

    def infos(self):
        """
        print all the arguments
        :return:
        """
        print 'content: ' + self.content_name + ' ' + str(self.content.shape)
        print 'style: ' + self.style_name + ' ' + str(self.style.shape)
        print 'start: ' + self.start
        print 'ratio: ' + str(self.ratio)
        print 'lenghts: ' + str(self.lengths)
        print 'style scale: ' + str(self.style_scale)
        print 'content_layer: ' + self.content_layer
        print 'style weights' + str(self.style_weights)
        print '\n'
        print 'optimization parameters: '
        for string in self.optimization.keys():
            print string + ' : ' + str(self.optimization[string])

def save(output,args=None):
    """
    save a deprocessed image
    :param output: deprocessed image
    :param args: args object for naming the file
    """
    if args is None:
        imsave('output.jpg',output)
    else:
        style = args.style_name
        content = args.content_name
        it = 'it'+str(args.optimization['maxiter'])
        c_max_len = 'c'+str(args.lengths)
        s_max_len = 's'+str(args.lengths * args.style_scale)
        ratio = 'r'+str(args.ratio)
        content_layer = args.content_layer
        style_layer = ''.join(args.style_weights.keys())

        string = 'outputs/' + '-'.join([style,s_max_len,content,c_max_len,it,content_layer,ratio,style_layer]) + '.jpg'

        imsave(string,output)
