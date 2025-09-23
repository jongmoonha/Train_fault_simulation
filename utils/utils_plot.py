from utils.utils_processing import fft_simple, filtering
from scipy.signal import hilbert
import numpy as np

def plot_raw(ax, t, x, title):
    ax.plot(t, x, lw=0.8)
    ax.set_title(title)

def plot_spectrum(ax, x, fs, title, vlines=None, xlim=None, ylim=None):
    f, A = fft_simple(x-np.mean(x), fs)
    ax.plot(f, A, lw=0.8)
    if vlines:
        for vf in vlines:
            ax.axvline(vf, ls='--', c='r', alpha=0.6)

    if xlim:
        ax.set_xlim(xlim)
    
    if ylim:
        ax.set_ylim([0, ylim])

def five_harmonics(base, n=5):
    return [k*base for k in range(1, n+1)]
