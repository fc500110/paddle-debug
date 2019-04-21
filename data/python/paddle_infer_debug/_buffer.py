# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals

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
        assert isinstance(data, list)
        return self.write(len(data), "!Q")

    def write_list(self, data, dtype='float32'):
        assert isinstance(data, list)
        #self.write(len(data), 'Q')
        self.write_size(data)
        fmt = {
            'float32': 'f',
            'int64': 'q',
            'int32': 'i',
            'size_t': 'Q'
        }.get(dtype)
        fmt = '!{}'.format(fmt)
        for v in data:
            self.write(v, fmt)
