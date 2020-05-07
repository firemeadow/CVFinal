import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compare_algs(windowed, pic, noise):
    algs = ['ssd', 'srd', 'sad', 'bgcsd', 'ncc']
    first = True
    for alg in algs:
        if windowed:
            temp = pd.read_csv("../data/" + pic + "/windowed_" + alg + ".csv", ',')
        else:
            print(alg)
            temp = pd.read_csv("../data/" + pic + "/" + alg + ".csv", ',')
        temp = temp.iloc[[noise]]
        if first:
            out = temp
            first = False
        else:
            out = pd.concat([out, temp])

    out.reset_index(drop=True, inplace=True)
    fig, axs = plt.subplots()
    if windowed:
        fig.suptitle("Windowed Algorithm Performance on Picture " + pic + ", and Noise Level " + str(noise))
    else:
        fig.suptitle("Algorithm Performance on Picture " + pic + ", and Noise Level " + str(noise))
    axs.scatter(out['u'], out['v'])

    i = 0
    for label in algs:
        axs.annotate(label,
                     (out.loc[i, 'u'], out.loc[i, 'v']),
                     xytext=(7, 0),
                     textcoords='offset points',
                     ha='left',
                     va='center')
        i += 1
    axs.grid(True, which='both')
    axs.axhline(y=0, color='k')
    axs.axvline(x=0, color='k')
    plt.xlabel("u")
    plt.ylabel("v")
    plt.show()


def compare_windowed(alg, pic, noise):
    first = True
    for i in range(2):
        if i:
            temp = pd.read_csv("../data/" + pic + "/" + alg + ".csv", ',')
        else:
            temp = pd.read_csv("../data/" + pic + "/windowed_" + alg + ".csv", ',')
        temp = temp.iloc[[noise]]
        if first:
            out = temp
            first = False
        else:
            out = pd.concat([out, temp])

    out.reset_index(drop=True, inplace=True)
    fig, axs = plt.subplots()
    fig.suptitle("Windowed vs Non-Windowed " + alg + " on Picture " + pic + ", and Noise Level " + str(noise))
    axs.scatter(out['u'], out['v'])

    i = 0
    labels = ['windowed', 'non-windowed']
    for label in labels:
        axs.annotate(label,
                     (out.loc[i, 'u'], out.loc[i, 'v']),
                     xytext=(7, 0),
                     textcoords='offset points',
                     ha='left',
                     va='center')
        i += 1
    axs.grid(True, which='both')
    axs.axhline(y=0, color='k')
    axs.axvline(x=0, color='k')
    plt.xlabel("u")
    plt.ylabel("v")
    plt.show()


def compare_pictures(alg, windowed, noise):
    pictures = ['aloe', 'baby1', 'baby2', 'plastic']
    first = True
    for pic in pictures:
        if not windowed:
            temp = pd.read_csv("../data/" + pic + "/" + alg + ".csv", ',')
        else:
            temp = pd.read_csv("../data/" + pic + "/windowed_" + alg + ".csv", ',')
        temp = temp.iloc[[noise]]
        if first:
            out = temp
            first = False
        else:
            out = pd.concat([out, temp])

    out.reset_index(drop=True, inplace=True)
    fig, axs = plt.subplots()
    if windowed:
        fig.suptitle("Windowed " + alg + " Performance across all Pictures, and Noise Level " + str(noise))
    else:
        fig.suptitle(alg + " Performance across all Pictures, and Noise Level " + str(noise))
    axs.scatter(out['u'], out['v'])

    i = 0
    for label in pictures:
        axs.annotate(label,
                     (out.loc[i, 'u'], out.loc[i, 'v']),
                     xytext=(7, 0),
                     textcoords='offset points',
                     ha='left',
                     va='center')
        i += 1
    axs.grid(True, which='both')
    axs.axhline(y=0, color='k')
    axs.axvline(x=0, color='k')
    plt.xlabel("u")
    plt.ylabel("v")
    plt.show()


def compare_noise(alg, windowed, pic):
    if not windowed:
        out = pd.read_csv("../data/" + pic + "/" + alg + ".csv", ',')
    else:
        out = pd.read_csv("../data/" + pic + "/windowed_" + alg + ".csv", ',')

    fig, axs = plt.subplots()
    if windowed:
        fig.suptitle("Windowed " + alg + " Performance on Picture " + pic + ", across all noise levels")
    else:
        fig.suptitle(alg + " Performance on Picture " + pic + ", across all noise levels")
    axs.scatter(out['u'], out['v'])

    i = 0
    labels = ['None', '1', '2', '4', '8', '16']
    for label in labels:
        axs.annotate(label,
                     (out.loc[i, 'u'], out.loc[i, 'v']),
                     xytext=(7, 0),
                     textcoords='offset points',
                     ha='left',
                     va='center')
        i += 1
    axs.grid(True, which='both')
    axs.axhline(y=0, color='k')
    axs.axvline(x=0, color='k')
    plt.xlabel("u")
    plt.ylabel("v")
    plt.show()
