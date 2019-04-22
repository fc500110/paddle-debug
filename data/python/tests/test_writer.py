#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import unittest
import numpy as np
import os
import random

from paddle_infer_debug import TensorInfo, Tensor, Unpacking, OfflineWriter


class TestOfflineWriter(unittest.TestCase):
    def setUp(self):
        self.x_info = TensorInfo('x', [-1, 2], dtype='int64', lod_level=0)
        self.y_info = TensorInfo('y', [-1, 10], dtype='float32', lod_level=1)
        self.x = Tensor(np.random.randint(100, size=2).tolist(), dtype='int64')
        self.y = Tensor(
            np.random.randn(10).tolist(), dtype='float32', lod=[[10]])

        self.filename = 'test_offline_writer.data.bin.{}'.format(
            random.randint(0, 100))
        self.assertFalse(os.path.exists(self.filename))
        self.writer = OfflineWriter(self.filename)

    def tearDown(self):
        os.system('rm -rf {}'.format(self.filename))
        self.writer.close()

    def test_write(self):
        self.writer.add_infos([self.x_info, self.y_info])
        self.writer.add_tensor(self.x)
        self.writer.add_tensor(self.y)
        self.writer.close()

        with open(self.filename, 'rb') as fr:
            buf = fr.read()
            unpack_data = Unpacking(buf)

        for info in [self.x_info, self.y_info]:
            unpack_info = unpack_data.get_tensor_info()
            self.assertEqual(info.name, unpack_info.name)
            self.assertEqual(info.shape, unpack_info.shape)
            self.assertEqual(info.dtype, unpack_info.dtype)
            self.assertEqual(info.lod_level, unpack_info.lod_level)

        for t in [self.x, self.y]:
            u = unpack_data.get_tensor()
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
