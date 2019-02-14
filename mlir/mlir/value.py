class Value():
    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype
        self.shape = shape

    def dump(self):
        return {
            b'name': self.name,
            b'dtype': self.dtype,
            b'shape': self.shape
        }
