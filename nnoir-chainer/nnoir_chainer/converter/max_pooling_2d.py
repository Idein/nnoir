import chainer.functions as F


class ConvertMaxPooling2D():

    def __init__(self, function, inputs, outputs):
        [kh, kw] = function.params['kernel']
        [sy, sx] = function.params['stride']
        [in_h, in_w] = inputs[0].shape[2:]
        [out_h, out_w] = outputs[0].shape[2:]
        [ph_pre, ph_post] = function.params['pad_h']
        [pw_pre, pw_post] = function.params['pad_w']
        ph_post = max(ph_post - ((ph_pre + in_h + ph_post) - ((out_h - 1) * sy + kh)), 0)
        pw_post = max(pw_post - ((pw_pre + in_w + pw_post) - ((out_w - 1) * sx + kw)), 0)
        padding = [(0, 0), (0, 0), (ph_pre, ph_post), (pw_pre, pw_post)]
        def pad(x): return F.pad(x, padding, mode='constant', constant_values=-float('inf'))
        self.f = lambda x: F.max_pooling_2d(pad(x),
                                            ksize=tuple(function.params['kernel']),
                                            stride=(sy, sx),
                                            pad=0)

    def __call__(self, x):
        return self.f(x)
