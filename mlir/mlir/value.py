class Value():
    def __init__(self, name, np_array=None, dtype=None, shape=None):
        self.name = name
        if np_array is None:
            self.dtype = dtype
            self.shape = shape
        else:
            self.dtype = np_array.dtype.str.encode()
            self.shape = np_array.shape

    def dump(self):
        return {
            b'name': self.name,
            b'dtype': self.dtype,
            b'shape': self.shape
        }
