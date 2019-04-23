from .paddle_tensor import TensorInfo, Tensor
from contextlib import contextmanager
import struct


class OfflineWriter(object):
    def __init__(self, filename):
        self.filename = filename
        self._file = open(self.filename, 'wb')

    def add_infos(self, infos):
        assert isinstance(infos, list)
        num_infos = len(infos)
        self._file.write(struct.pack("!Q", num_infos))
        for info in infos:
            assert isinstance(info, TensorInfo)
            self._file.write(info.pack())

    def add_tensor(self, tensor):
        assert isinstance(tensor, Tensor)
        self._file.write(tensor.pack())

    def close(self):
        self._file.close()


@contextmanager
def get_writer(filename):
    writer = OfflineWriter(filename)
    yield writer
    writer.close()
