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
### This file was copied from
# https://github.com/NVIDIA-Jetson/tf_trt_models/blob/master/tf_trt_models/graph_utils.py
###

import tensorflow as tf


def make_const6(const6_name='const6'):
    graph = tf.Graph()
    with graph.as_default():
        tf_6 = tf.constant(dtype=tf.float32, value=6.0, name=const6_name)
    return graph.as_graph_def()


def make_relu6(output_name, input_name, const6_name='const6'):
    graph = tf.Graph()
    with graph.as_default():
        tf_x = tf.placeholder(tf.float32, [10, 10], name=input_name)
        tf_6 = tf.constant(dtype=tf.float32, value=6.0, name=const6_name)
        with tf.name_scope(output_name):
            tf_y1 = tf.nn.relu(tf_x, name='relu1')
            tf_y2 = tf.nn.relu(tf.subtract(tf_x, tf_6, name='sub1'), name='relu2')

            #tf_y = tf.nn.relu(tf.subtract(tf_6, tf.nn.relu(tf_x, name='relu1'), name='sub'), name='relu2')
        #tf_y = tf.subtract(tf_6, tf_y, name=output_name)
        tf_y = tf.subtract(tf_y1, tf_y2, name=output_name)
        
    graph_def = graph.as_graph_def()
    graph_def.node[-1].name = output_name

    # remove unused nodes
    for node in graph_def.node:
        if node.name == input_name:
            graph_def.node.remove(node)
    for node in graph_def.node:
        if node.name == const6_name:
            graph_def.node.remove(node)
    for node in graph_def.node:
        if node.op == '_Neg':
            node.op = 'Neg'
            
    return graph_def


def convert_relu6(graph_def, const6_name='const6'):
    # add constant 6
    has_const6 = False
    for node in graph_def.node:
        if node.name == const6_name:
            has_const6 = True
    if not has_const6:
        const6_graph_def = make_const6(const6_name=const6_name)
        graph_def.node.extend(const6_graph_def.node)
        
    for node in graph_def.node:
        if node.op == 'Relu6':
            input_name = node.input[0]
            output_name = node.name
            relu6_graph_def = make_relu6(output_name, input_name, const6_name=const6_name)
            graph_def.node.remove(node)
            graph_def.node.extend(relu6_graph_def.node)
            
    return graph_def


def remove_node(graph_def, node):
    for n in graph_def.node:
        if node.name in n.input:
            n.input.remove(node.name)
        ctrl_name = '^' + node.name
        if ctrl_name in n.input:
            n.input.remove(ctrl_name)
    graph_def.node.remove(node)


def remove_op(graph_def, op_name):
    matches = [node for node in graph_def.node if node.op == op_name]
    for match in matches:
        remove_node(graph_def, match)
