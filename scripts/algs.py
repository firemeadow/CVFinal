import numpy as np
import matplotlib.pyplot as plt


def cgauss(i, j):
    out = (i ** 2) + (j ** 2)
    out = np.exp(-out / 2)
    out /= (2 * np.pi)
    return out


def make_window(shape, xi, yi, im0_x_shape, im0_y_shape, fill):
    out = np.zeros(shape)
    x_size, y_size = shape
    if x_size % 2 == 0:
        x_mid = int(x_size / 2)
    else:
        x_mid = int((x_size - 1) / 2)

    if y_size % 2 == 0:
        y_mid = int(y_size / 2)
    else:
        y_mid = int((y_size - 1) / 2)

    patch_size = (x_size + y_size) / 2
    scale = 5 / patch_size

    for i in range(x_size):
        for j in range(y_size):
            relative_x = xi - x_mid + i
            relative_y = yi - y_mid + j
            if relative_x < 0 or relative_x > im0_x_shape or relative_y < 0 or relative_y > im0_y_shape:
                out[i, j] = fill
            else:
                out[i, j] = cgauss((i - x_mid) * scale, (j - y_mid) * scale)
    return out


def ssd(diff):
    return np.sum(diff**2)


def srd(diff, delta):
    sum = 0
    for i in range(3):
        if np.abs(diff[i]) <= delta:
            val = (diff[i]**2) / 2
        else:
            val = (np.abs(diff[i]) - (delta/2)) * delta
        sum += val
    return sum


def sad(diff):
    return np.sum(np.abs(diff))


def find_bg(i0, i1):
    std0 = np.std(i0, axis=(0, 1))
    std1 = np.std(i1, axis=(0, 1))
    mean0 = np.mean(i0, axis=(0, 1))
    mean1 = np.mean(i1, axis=(0, 1))
    gain = np.divide(std1, std0)
    bias = np.subtract(mean1, np.multiply(mean0, gain))
    return bias, gain


def bgcsd(diff, pix0, bias, gain):
    out = np.add(np.multiply(pix0, gain), bias)
    out = np.power(np.subtract(out, diff), 2)
    return np.sum(out)


def ncc(pix0, pix1, mean0, mean1, std0, std1):
    pix0_norm = np.subtract(pix0, mean0)
    pix0_norm = np.divide(pix0_norm, std0)
    pix1_norm = np.subtract(pix1, mean1)
    pix1_norm = np.divide(pix1_norm, std1)
    out = np.multiply(pix0_norm, pix1_norm)
    return np.sum(out)


def detect_motion(im0, im1, x, y, patch_size, alg, delta=1, windowed=False):
    if patch_size % 2 != 1:
        print("Error: Patch Size Must Be Odd.")
        exit(0)

    patch_mid = int((patch_size - 1) / 2)
    pix0 = im0[x, y, :]
    min_val = np.inf
    max_val = -np.inf

    if alg == 'ncc':
        mean0 = np.mean(im0, (0, 1))
        mean1 = np.mean(im1, (0, 1))
        std0 = np.std(im0, (0, 1))
        std1 = np.std(im0, (0, 1))
        if windowed:
            window = make_window((patch_size, patch_size), x, y, im0.shape[0], im0.shape[1], 0)
    elif windowed:
        window = make_window((patch_size, patch_size), x, y, im0.shape[0], im0.shape[1], 999)

    if alg == 'bgcsd':
        bias, gain = find_bg(im0, im1)

    for i in range(patch_size):
        for j in range(patch_size):
            xi = x - patch_mid + i
            yi = y - patch_mid + j

            if xi < 0 or xi >= im0.shape[0]:
                continue
            elif yi < 0 or yi >= im0.shape[1]:
                continue
            else:
                pix1 = im1[xi, yi, :]
                diff = np.subtract(pix1, pix0)
                if alg == 'ssd':
                    val = ssd(diff)

                elif alg == 'srd':
                    val = srd(diff, delta)

                elif alg == 'sad':
                    val = sad(diff)

                elif alg == 'bgcsd':
                    val = bgcsd(diff, pix0, bias, gain)

                elif alg == 'ncc':
                    val = ncc(pix0, pix1, mean0, mean1, std0, std1)

                if windowed:
                    if alg == 'ncc':
                        val *= window[i, j]
                    else:
                        val *= 1 - window[i, j]

                if alg == 'ncc':
                    if val > max_val:
                        max_val = val
                        u = xi - x
                        v = yi - y
                else:
                    if val < min_val:
                        min_val = val
                        u = xi - x
                        v = yi - y
    return u, v


