import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import algs


def run_alg(pic_name, start_pic, end_pic, alg_name, n_points, patch_size, windowed=False):
    im0 = imread("../data/" + pic_name + "/view" + start_pic + ".png")
    im1 = imread("../data/" + pic_name + "/view" + end_pic + ".png")
    u_arr = []
    v_arr = []
    random_x = np.random.choice(np.arange(im0.shape[0]), n_points)
    random_y = np.random.choice(np.arange(im0.shape[1]), n_points)

    for x, y in zip(random_x, random_y):
        # print("Point: " + str(x) + ", " + str(y))
        u, v = algs.detect_motion(im0, im1, x, y, patch_size, alg_name)
        u_arr.append(u)
        v_arr.append(v)
        # print("Predicted shift: " + str(u) + ", " + str(v))

    u_arr = np.array(u_arr)
    v_arr = np.array(v_arr)
    ui = np.mean(u_arr)
    vi = np.mean(v_arr)

    return ui, vi


if __name__ == "__main__":
    picture = 'baby1'
    start = 0
    end = 1
    algorithm_name = 'ncc'
    num_points = 500
    patch_size = 21
    algs = ['ssd', 'srd', 'sad', 'bgscsd', 'ncc']
    for i in range(1):
        for alg in algs:
            u, v = run_alg(picture, start, end, algorithm_name, num_points, patch_size, windowed=bool(i))
