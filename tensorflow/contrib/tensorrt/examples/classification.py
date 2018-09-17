# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of NVIDIA CORPORATION nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================

import os
import subprocess
import tensorflow as tf
import nets.nets_factory
import tensorflow.contrib.slim as slim
import official.resnet.imagenet_main
from preprocessing import inception_preprocessing, vgg_preprocessing

from graph_utils import convert_relu6

class NetDef:
    """Contains definition of a model"""
    def __init__(self, name, url=None, checkpoint_name=None, preprocess='inception',
            input_size=224, slim=True, postprocess=tf.nn.softmax, model=None, num_classes=1001):
        self.name = name
        self.url = url
        self.checkpoint_name = checkpoint_name
        if preprocess == 'inception':
            self.preprocess = inception_preprocessing.preprocess_image
        elif preprocess == 'vgg':
            self.preprocess = vgg_preprocessing.preprocess_image
        self.input_width = input_size
        self.input_height = input_size
        self.slim = slim
        self.postprocess = postprocess
        self.model = model
        self.num_classes = num_classes

    def get_input_dims(self):
        return self.input_width, self.input_height

    def get_num_classes(self):
        return self.num_classes

def get_netdef(model):
    """
    Creates the dictionary NETS with model names as keys and NetDef as values.
    Returns the NetDef corresponding to the model specified in the parameter.

    model: string, the model name (see NETS table)
    """
    NETS = {
        'mobilenet_v1': NetDef(
            name='mobilenet_v1',
            url='http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz',
            checkpoint_name='mobilenet_v1_1.0_224.ckpt'),

        'mobilenet_v2': NetDef(
            name='mobilenet_v2_140',
            url='https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz',
            checkpoint_name='mobilenet_v2_1.4_224.ckpt'),

        'nasnet_mobile': NetDef(
            name='nasnet_mobile',
            url='https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz',
            checkpoint_name='model.ckpt'),

        'nasnet_large': NetDef(
            name='nasnet_large',
            url='https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz',
            checkpoint_name='model.ckpt',
            input_size=331),

        'resnet_v1_50': NetDef(
            name='resnet_v1_50',
            url='http://download.tensorflow.org/models/official/20180601_resnet_v1_imagenet_checkpoint.tar.gz',
            checkpoint_name=os.path.join('20180601_resnet_v1_imagenet_checkpoint', 'model.ckpt-257706'),
            slim=False,
            preprocess='vgg',
            model=official.resnet.imagenet_main.ImagenetModel(resnet_size=50, resnet_version=1)),

        'resnet_v2_50': NetDef(
            name='resnet_v2_50',
            url='http://download.tensorflow.org/models/official/20180601_resnet_v2_imagenet_checkpoint.tar.gz',
            checkpoint_name=os.path.join('20180601_resnet_v2_imagenet_checkpoint', 'model.ckpt-258931'),
            slim=False,
            preprocess='vgg',
            model=official.resnet.imagenet_main.ImagenetModel(resnet_size=50, resnet_version=2)),

        'vgg_16': NetDef(
            name='vgg_16',
            url='http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
            checkpoint_name='vgg_16.ckpt',
            preprocess='vgg',
            num_classes=1000),

        'vgg_19': NetDef(
            name='vgg_19',
            url='http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
            checkpoint_name='vgg_19.ckpt',
            preprocess='vgg',
            num_classes=1000),

        'inception_v3': NetDef(
            name='inception_v3',
            url='http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
            checkpoint_name='inception_v3.ckpt',
            input_size=299),

        'inception_v4': NetDef(
            name='inception_v4',
            url='http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
            checkpoint_name='inception_v4.ckpt',
            input_size=299),
    }
    return NETS[model]

def _deserialize_image_record(record):
    feature_map = {
        'image/encoded':          tf.FixedLenFeature([ ], tf.string, ''),
        'image/class/label':      tf.FixedLenFeature([1], tf.int64,  -1),
        'image/class/text':       tf.FixedLenFeature([ ], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label   = tf.cast(obj['image/class/label'], tf.int32)
        bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                            for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
        text    = obj['image/class/text']
        return imgdata, label, bbox, text

def get_preprocess_fn(model, mode='classification'):
    """Creates a function to parse and process a TFRecord using the model's parameters

    model: string, the model name (see NETS table)
    mode: string, whether the model is for classification or detection
    returns: function, the preprocessing function for a record
    """
    def process(record):
        # Parse TFRecord
        imgdata, label, bbox, text = _deserialize_image_record(record)
        label -= 1 # Change to 0-based (don't use background class)
        try:    image = tf.image.decode_jpeg(imgdata, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
        except: image = tf.image.decode_png(imgdata, channels=3)
        # Use model's preprocessing function
        netdef = get_netdef(model)
        image = netdef.preprocess(image, netdef.input_height, netdef.input_width, is_training=False)
        return image, label

    return process

def build_classification_graph(model, download_dir='./data'):
    """Builds an image classification model by name

    This function builds an image classification model given a model
    name, parameter checkpoint file path, and number of classes.  This
    function performs some graph processing (such as replacing relu6(x)
    operations with relu(x) - relu(x-6)) to produce a graph that is
    well optimized by the TensorRT package in TensorFlow 1.7+.

    model: string, the model name (see NETS table)
    download_dir: directory to store downloaded model checkpoints
    returns: tensorflow.GraphDef, the TensorRT compatible frozen graph
    """
    netdef = get_netdef(model)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
            tf_input = tf.placeholder(tf.float32, [None, netdef.input_height, netdef.input_width, 3], name='input')
            if netdef.slim:
                # TF Slim Model: get model function from nets_factory
                network_fn = nets.nets_factory.get_network_fn(netdef.name, netdef.num_classes,
                        is_training=False)
                tf_net, tf_end_points = network_fn(tf_input)
            else:
                # TF Official Model: get model function from NETS
                tf_net = netdef.model(tf_input, training=False)

            tf_output = tf.identity(tf_net, name='logits')
            num_classes = tf_output.get_shape().as_list()[1]
            if num_classes == 1001:
                # Shift class down by 1 if background class was included
                tf_output_classes = tf.add(tf.argmax(tf_output, axis=1), -1, name='classes')
            else:
                tf_output_classes = tf.argmax(tf_output, axis=1, name='classes')

            # download checkpoint
            checkpoint_path = download_classification_checkpoint(model, download_dir)
            # load checkpoint
            tf_saver = tf.train.Saver()
            tf_saver.restore(save_path=checkpoint_path, sess=tf_sess)

            # freeze graph
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                tf_sess,
                tf_sess.graph_def,
                output_node_names=['logits', 'classes']
            )

            # remove relu 6
            frozen_graph = convert_relu6(frozen_graph)

    return frozen_graph

def download_classification_checkpoint(model, output_dir='.'):
    """Downloads an image classification model pretrained checkpoint by name

    model: string, the model name (see NETS table)
    output_dir: string, the directory where files are downloaded to
    returns: string, path to the checkpoint file containing trained model params
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    modeldir_path = os.path.join(output_dir, model)
    if not os.path.exists(modeldir_path):
        os.makedirs(modeldir_path)

    # Build path to checkpoint and see if we have it already
    checkpoint_path = os.path.join(modeldir_path, get_netdef(model).checkpoint_name)
    if os.path.exists(os.path.dirname(checkpoint_path)):
        for filename in os.listdir(os.path.dirname(checkpoint_path)):
            if filename.startswith(os.path.basename(checkpoint_path)):
                print('Using checkpoint found at:', checkpoint_path)
                return checkpoint_path
  
    # We don't have the checkpoint, need to redownload or extract
    print('Downloading checkpoint - not found at:', checkpoint_path)
    modeltar_path = os.path.join(output_dir, os.path.basename(get_netdef(model).url))
    if not os.path.isfile(modeltar_path):
        subprocess.call(['wget', '--no-check-certificate', get_netdef(model).url, '-O', modeltar_path])
    subprocess.call(['tar', '-xzf', modeltar_path, '-C', modeldir_path])

    return checkpoint_path
