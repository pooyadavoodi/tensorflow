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

import argparse
import ast
import sys
import re


def parse_file(filename):
    with open(filename) as f:
        f_data = f.read()
    results = {}
    def regex_match(regex):
        match = re.match(regex, f_data, re.DOTALL)
        if match is not None:
            results[match.group(1)] = match.group(2)
    regex_match('.*(model): (\w*)')
    regex_match('.*(accuracy): (\d*\.\d*)')
    assert len(results) == 2, '{}'.format(results)
    return results

def check_accuracy(res, tol):
    dest = {
        'mobilenet_v1': 71.02,
        'mobilenet_v2': 74.11,
        'nasnet_large': 82.71,
        'nasnet_mobile': 73.97,
        'resnet_v1_50': 75.91,
        'resnet_v2_50': 76.05,
        'vgg_16': 70.89,
        'vgg_19': 71.00,
        'inception_v3': 77.98,
        'inception_v4': 80.18,
    }
    if abs(float(res['accuracy']) - dest[res['model']]) < tol:
        print("PASS")
    else:
        print("FAIL: accuracy {} vs. {}".format(res['accuracy'], dest[res['model']]))
        sys.exit(1)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input')
    parser.add_argument('--tolerance', dest='tolerance', type=float, default=0.1)
    
    args = parser.parse_args()
    filename = args.input
    tolerance = args.tolerance

    print()
    print('checking accuracy...')
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))
    check_accuracy(parse_file(filename), tolerance)

