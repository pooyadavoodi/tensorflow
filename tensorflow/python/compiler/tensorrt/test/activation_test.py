# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic tests for TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class ActivationTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, x):
    x = nn.relu(x, name="relu1")
    x = nn.relu(x, name="relu2")
    return array_ops.identity(x, name="output_0")

  def GetParams(self):
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 3, 6, 6]],
                            [[100, 3, 6, 6]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"TRTEngineOp_0": ["relu1", "relu2"]}

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    return not trt_test.IsQuantizationMode(run_params.precision_mode) and \
           not run_params.dynamic_engine

class ActivationDynamicShapeTest(trt_test.TfTrtIntegrationTestBase):

  def GraphFn(self, x):
    x = nn.relu(x, name="relu1")
    x = nn.relu(x, name="relu2")
    return array_ops.identity(x, name="output")

  def GetParams(self):
    # TODO(tmorris): Remove env vars when calibration is done.
    os.environ["TFTRT_OPT_PROFILE_MIN"] = "1,2,1,1"
    os.environ["TFTRT_OPT_PROFILE_OPT"] = "1,2,3,5"
    os.environ["TFTRT_OPT_PROFILE_MAX"] = "8,2,5,5"

    input_dims = [[[1, 2, 3, 5]], [[8, 2, 5, 1]]]
    expected_output_dims = [[[1, 2, 3, 5]], [[8, 2, 5, 1]]]
    return trt_test.TfTrtIntegrationTestParams(
        graph_fn=self.GraphFn,
        input_specs=[
            tensor_spec.TensorSpec([None, 2, None, None], dtypes.float32,
                                   "input")
        ],
        output_specs=[
            tensor_spec.TensorSpec([None, 2, None, None], dtypes.float32,
                                   "output")
        ],
        input_dims=input_dims,
        expected_output_dims=expected_output_dims)

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {"TRTEngineOp_0": ["relu1", "relu2"]}

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    # TODO(tmorris): Enable in TRT 6.0
    return not trt_test.IsQuantizationMode(run_params.precision_mode) and \
           not run_params.dynamic_engine and run_params.precision_mode == "FP32"

if __name__ == "__main__":
  test.main()
