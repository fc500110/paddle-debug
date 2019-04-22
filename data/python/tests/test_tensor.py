#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import unittest
import numpy as np
import six
import struct

from paddle_infer_debug import TensorInfo, Tensor, Unpacking


class TestTensor(unittest.TestCase):
    def setUp(self):
        self.x_info = TensorInfo('x', [-1, 2], dtype='int64', lod_level=0)
        self.y_info = TensorInfo('y', [-1, 10], dtype='float32', lod_level=1)
        self.x = Tensor(np.random.randint(100, size=2).tolist(), dtype='int64')
        self.y = Tensor(
            np.random.randn(10).tolist(), dtype='float32', lod=[[10]])

        class NumberPackage(object):
            def __init__(self, value, fmt):
                self.data = struct.pack(fmt, value)

            def pack(self):
                return self.data

        num_inputs = NumberPackage(2, '!Q')
        package = [
            _.pack()
            for _ in [num_inputs, self.x_info, self.y_info, self.x, self.y]
        ]
        self.data = Unpacking(six.b("").join(package))

        self.unpack_infos = [
            self.data.get_tensor_info() for i in range(self.data.num_info)
        ]
        num_sample = 2
        self.unpack_tensors = [
            self.data.get_tensor() for i in range(num_sample)
        ]

    def test_tensor_info(self):
        infos = [self.x_info, self.y_info]
        self.assertEqual(len(infos), len(self.unpack_infos))
        for info, unpack_info in zip(infos, self.unpack_infos):
            self.assertEqual(info.name, unpack_info.name)
            self.assertEqual(info.shape, unpack_info.shape)
            self.assertEqual(info.dtype, unpack_info.dtype)
            self.assertEqual(info.lod_level, unpack_info.lod_level)

    def test_tensor(self):
        tensors = [self.x, self.y]
        self.assertEqual(len(tensors), len(self.unpack_tensors))
        for t, u in zip(tensors, self.unpack_tensors):
            self.assertEqual(t.dtype, u.dtype)
            self.assertEqual(t.lod, u.lod)
            if t.dtype == 'float32':
                self.assertEqual(len(t.data), len(u.data))
                for a, b in zip(t.data, u.data):
                    self.assertAlmostEqual(a, b, 6)
            else:
                self.assertEqual(t.data, u.data)


if __name__ == "__main__":
    unittest.main()
