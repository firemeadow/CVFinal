import numpy as np
from imageio import imread
import pandas as pd
import matplotlib.pyplot as plt
import algs


def run_alg(pic_name, start_pic, end_pic, alg_name, n_points, patch_size, noise=None, windowed=False):
    im0 = imread("../data/" + pic_name + "/view" + str(start_pic) + ".png")
    im1 = imread("../data/" + pic_name + "/view" + str(end_pic) + ".png")
    u_arr = []
    v_arr = []
    random_x = np.random.choice(np.arange(im0.shape[0]), n_points)
    random_y = np.random.choice(np.arange(im0.shape[1]), n_points)

    if noise is not None:
        im0 = np.add(im0, np.round(np.random.normal(0, noise, np.shape(im0)), 0).astype(int))
        im1 = np.add(im1, np.round(np.random.normal(0, noise, np.shape(im1)), 0).astype(int))

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


# ['ssd', 'srd', 'sad', 'bgscsd', 'ncc']
if __name__ == "__main__":
    pictures = ['baby1', 'baby2', 'aloe', 'plastic']
    start = 0
    end = 1
    num_points = 1000
    patch_size = 51
    algorithms = ['ssd', 'srd', 'sad', 'bgcsd', 'ncc']
    noise_param = [None, 1, 2, 4, 8, 16]
    for picture in pictures:
        print("Picture: " + picture)
        for alg in algorithms:
            print("Algorithm: " + alg)
            for i in range(2):
                a = []
                b = []
                for j in noise_param:
                    if j is not None:
                        u, v = run_alg(picture, start, end, alg, num_points, patch_size, j, windowed=bool(i))
                    else:
                        u, v = run_alg(picture, start, end, alg, num_points, patch_size, windowed=bool(i))
                    a.append(u)
                    b.append(v)
                a = pd.DataFrame(np.array(a), columns=['u'])
                b = pd.DataFrame(np.array(b), columns=['v'])
                out = pd.concat([a, b], axis=1)
                if i:
                    out.to_csv("../data/" + picture + "/windowed_ " + alg + ".csv", ',', index=False)
                else:
                    out.to_csv("../data/" + picture + "/" + alg + ".csv", ',', index=False)
