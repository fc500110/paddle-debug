# -*- coding: utf-8 -*-

import six
import struct

if six.PY3:
    long = int


class Buffer(object):
    def __init__(self):
        self._buffer = six.b("")

    @property
    def data(self):
        return self._buffer

    def write(self, data, fmt=None):
        if fmt is None:
            dtype = type(data)
            if dtype is int:
                fmt = '!i'
            elif dtype is long:
                fmt = '!l'
            elif dtype is float:
                fmt = '!f'
            elif dtype is str:
                data = data.encode('utf-8')
                fmt = '!{:d}s'.format(len(data))
                self._buffer = six.b("").join(
                    [self._buffer, struct.pack('!i', len(data))])
            elif dtype is bytes:
                fmt = '!{:d}s'.format(len(data))
                self._buffer = six.b("").join(
                    [self._buffer, struct.pack('!i', len(data))])
            else:
                pass

        self._buffer = six.b("").join([self._buffer, struct.pack(fmt, data)])

    def write_size(self, data):
        pass

    def write_list(self, data):
        pass


class Header(object):
    def __init__(self, inputs):
        self._buffer = Buffer()
        self.__add_inputs(inputs)

    def __add_inputs(self, inputs):
        self._buffer.write(len(inputs))

        for info in inputs:
            if isinstance(info, list):
                name, shape, dtype, level = info
            if isinstance(info, dict):
                name = info.get('name', '')
                shape = info['shape']
                dtype = info['dtype']
                level = info.get('level', 0)

            assert isinstance(name, str)
            assert isinstance(shape, list)
            assert isinstance(dtype, str)
            assert isinstance(level, int)
            #print(name, shape, dtype, level)

            self.__add_name(name)
            self.__add_shape(shape)
            self.__add_dtype(dtype)
            self.__add_lod_level(level)

    def __add_name(self, name):
        self._buffer.write(name)

    def __add_shape(self, shape):
        self._buffer.write(len(shape), '!I')
        for d in shape:
            self._buffer.write(d)

    def __add_dtype(self, dtype):
        dtype = dtype.lower()
        assert dtype in ['float32', 'int64', 'int32']
        self._buffer.write(dtype)

    def __add_lod_level(self, level):
        self._buffer.write(level)

    def pack(self):
        return self._buffer.data


class Tensor(object):
    def __init__(self, data, dtype, lod=None):
        self._buffer = Buffer()
        if lod:
            self.__add_lod(lod)
        self._fmt = '!' + {
            "INT64": 'l',
            "INT32": 'i',
            "FLOAT32": 'f'
        }.get(dtype.upper())
        self._buffer.write(len(data))
        for v in data:
            self._buffer.write(v, fmt=self._fmt)

    def __add_lod(self, lod):
        self._buffer.write(len(lod), fmt='!I')
        for v in value:
            self._buffer.write(len(v))
            for i in v:
                self._buffer.write(i, '!q')

    def pack(self):
        return self._buffer.data


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
