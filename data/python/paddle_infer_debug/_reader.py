from contextlib import contextmanager

class OfflineReader(object):
    def __init__(self, filename):
        self.filename = filename
        self._file = open(filename, 'rb')

    def get(self):
        pass

    def get_tensor_info(self):
        pass

    def get_tensor(self):
        pass

    def close(self):
        self._file.close()


@contextmanager
def get_reader(filename):
    reader = OfflineReader(filenmae)
    yield reader
    reader.close()
