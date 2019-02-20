import numpy as np


def get_conv_outsize(size, k, s, pre_p, post_p, cover_all=False, d=1):
    dk = k + (k - 1) * (d - 1)
    return (size + pre_p + post_p - dk) // s + 1


def im2col_cpu(img, kernel, stride, ph, pw, dilate=(1, 1), pval=0, cover_all=False):
    n, c, h, w = img.shape
    kh, kw = kernel
    sy, sx = stride
    dy, dx = dilate
    pre_ph = ph[0]
    post_ph = ph[1]
    pre_pw = pw[0]
    post_pw = pw[1]
    out_h = get_conv_outsize(h, kh, sy, pre_ph, post_ph, cover_all, dy)
    out_w = get_conv_outsize(w, kw, sx, pre_pw, post_pw, cover_all, dx)

    img = np.pad(img,
                 ((0, 0), (0, 0), (pre_ph, post_ph), (pre_pw, post_pw)),
                 mode='constant', constant_values=(pval,))
    col = np.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]

    return img, col


def col2im_cpu(col, stride, ph, pw, h, w, dy=1, dx=1):
    sy, sx = stride
    n, c, kh, kw, out_h, out_w = col.shape
    img = np.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1), dtype=col.dtype)
    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            img[:, :, jdy:j_lim:sy, idx:i_lim:sx] += col[:, :, j, i]
    return img[:, :, ph:h + ph, pw:w + pw]
