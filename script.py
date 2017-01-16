#!/usr/bin/env python
from Transfer import *
import time
from collections import OrderedDict
from features_extraction import *

net, transformer = build_net()
caffe.set_mode_gpu()

net, transformer = build_net()
caffe.set_mode_gpu()

args = args(net, transformer)
# # Visualize parameters
# args.infos()

args.change_content('tubingen')
args.change_style('starry_night')
args.start = 'content'
args.content_layer = 'conv5_1'
args.style_scale = 1.
args.ratio = 0
args.lengths = {'content' : 500, 'style' : 300}
args.optimization['maxiter'] = 200
args.style_weights = OrderedDict([('conv1_1', 1.),
                                 ('conv2_1', 1.),
                                 ('conv3_1', 1.),
                                 ('conv4_1', 1.),
                                 ('conv5_1', 1.)])
output = transfer(*args.get())
print_image([args.content,args.style,output])
