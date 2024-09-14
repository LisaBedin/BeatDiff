import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = '/mnt/data/lisa/ecg_results/inpainting_10s'

    npz = np.load(os.path.join(file_path, 'limbs_V2_V4.npz'))
    gen_beat = npz['posterior'][10]
    real_beat = npz['real']
    gen_beat /= np.max(np.absolute(gen_beat), axis=1)[:, np.newaxis]
    real_beat /= np.max(np.absolute(real_beat), axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(1, 1, figsize=(25, 8))
    fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
    for i, (track, track_real) in enumerate(zip(gen_beat, real_beat)):
        ax.plot(track - i, color='blue', alpha=.7, lw=3)
        ax.plot(track_real - i, color='red', alpha=.7, lw=3)
    fig.savefig(os.path.join(file_path, 'limbs_V2_V4.pdf'))
    plt.show()

    npz = np.load(os.path.join(file_path, 'limbs_V2_V4_AF.npz'))
    gen_beat = npz['posterior'][30]
    real_beat = npz['real']
    gen_beat /= np.max(np.absolute(gen_beat), axis=1)[:, np.newaxis]
    real_beat /= np.max(np.absolute(real_beat), axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(1, 1, figsize=(25, 8))
    fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
    for i, (track, track_real) in enumerate(zip(gen_beat, real_beat)):
        ax.plot(track - i, color='blue', alpha=.7, lw=3)
        ax.plot(track_real - i, color='red', alpha=.7, lw=3)
    fig.savefig(os.path.join(file_path, 'limbs_V2_V4_AF.pdf'))
    plt.show()
