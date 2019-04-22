# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals

__all__ = ["TensorInfo", "Tensor", "Unpacking"]

import six
import struct

from ._buffer import Buffer

if six.PY3:
    long = int


class Package(object):
    def __init__(self, need_persist=True):
        self._buffer = Buffer()
        self.need_persist = need_persist

    def pack(self):
        return self._buffer.data if self.need_persist else None

    def unpack(self):
        pass


class TensorInfo(Package):
    def __init__(self, name, shape, dtype, lod_level=0, need_persist=True):
        super(TensorInfo, self).__init__(need_persist)
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.lod_level = lod_level

        self.need_persist = need_persist

        if self.need_persist:
            self.__add_name(name)
            self.__add_shape(shape)
            self.__add_dtype(dtype)
            self.__add_lod_level(lod_level)

    def __str__(self):
        return str(
            dict(
                name=self.name,
                shape=self.shape,
                dtype=self.dtype,
                lod_level=self.lod_level))

    def __repr__(self):
        return str(self)

    def __add_name(self, name):
        assert isinstance(name, str)
        self._buffer.write(name)

    def __add_shape(self, shape):
        assert isinstance(shape, list)
        assert shape[0] == -1
        self._buffer.write_list(shape, dtype='int32')

    def __add_dtype(self, dtype):
        assert isinstance(dtype, str)
        dtype = dtype.lower()
        assert dtype in ['float32', 'int64', 'int32']
        self._buffer.write(dtype)

    def __add_lod_level(self, level):
        assert isinstance(level, int)
        self._buffer.write(level)


class Tensor(Package):
    def __init__(self, data, dtype, lod=None, need_persist=True):
        super(Tensor, self).__init__(need_persist)
        self.data = data
        self.dtype = dtype
        self.lod = lod if lod else []
        self.need_persist = need_persist
        if self.need_persist:
            if lod:
                self.__add_lod(lod)
            self.__add_data(data, dtype)

    def __str__(self):
        return str(dict(data=self.data, dtype=self.dtype, lod=self.lod))

    def __repr__(self):
        return str(self)

    def __add_lod(self, lod):
        for v in lod:
            self._buffer.write_list(v, dtype='size_t')

    def __add_data(self, data, dtype):
        self._buffer.write_list(data, dtype)


class Unpacking(object):
    def __init__(self, buffer):
        self._buffer = buffer
        self._offset = 0
        self._dtypes = []
        self._levels = []
        #self.num_package = -1
        self._i_package = 0
        self.num_info = self._get_size()
        self._i_info = 0

    def __get(self, fmt):
        value = struct.unpack_from(fmt, self._buffer[self._offset:])[0]
        self._offset += struct.calcsize(fmt)
        return value

    def _get_str(self):
        size = self.__get("!i")
        value = self.__get("!{}s".format(size)).decode('utf-8')
        return value

    def _get_size(self):
        return self.__get("!Q")

    def _get_list(self, fmt):
        size = self._get_size()
        value = []
        for i in range(size):
            value.append(self.__get(fmt))
        return value

    def get_tensor_info(self):
        self._i_info += 1
        name = self._get_str()
        shape = self._get_list('!i')
        dtype = self._get_str()
        lod_level = self.__get('!i')
        self._dtypes.append(dtype)
        self._levels.append(lod_level)
        #if self._i_info == self.num_info:
        #    self.num_package = self._get_size()
        return TensorInfo(name, shape, dtype, lod_level, need_persist=False)

    def get_tensor(self):
        dtype = self._dtypes[self._i_package % self.num_info]
        lod_level = self._levels[self._i_package % self.num_info]
        self._i_package += 1
        lod = []
        for i in range(lod_level):
            lod.append(self._get_list('!Q'))
        fmt = {"int64": 'q', "int32": 'i', "float32": 'f'}.get(dtype)
        fmt = '!' + fmt
        data = self._get_list(fmt)
        return Tensor(data, dtype, lod, need_persist=False)


def main():
    # name, shape, dtype, level
    inputs = [("x", [-1, 10], 'float32', 0), ("y", [2], 'int64', 0)]
    header = Header(inputs)
    buf = header.pack()

    offset = 0

    def unpack(fmt):
        nonlocal offset, buf
        r = struct.unpack_from(fmt, buf[offset:])
        offset += struct.calcsize(fmt)
        return r

    num_inputs = unpack('!I')[0]

    print(num_inputs)
    for i in range(num_inputs):
        name = unpack("!{unpack('!I')s}")
        print(name)

    return

    size = struct.unpack_from('!I', buf)[0]
    print(size)
    offset = struct.calcsize('!I')
    size = struct.unpack_from('!I', buf[offset:])[0]
    offset += struct.calcsize('!I')
    name = struct.unpack_from(f'!{size}s', buf[offset:])[0]
    print(name)
    size = struct.unpack_from('!I', buf[offset:])[0]
    offset += struct.calcsize('!I')
    shape = struct.unpack_from(f'{size}i', buf[offset:])
    print(shape)


if __name__ == "__main__":
    main()
