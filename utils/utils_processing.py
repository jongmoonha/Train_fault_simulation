import numpy as np
from scipy import signal, stats
from scipy.interpolate import interp1d
from matplotlib import pyplot
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import scipy.stats
from utils import kurtogram as kg

def fft_simple(xn,fs):
  N=len(xn)
  T=N/fs
  f=np.arange(0,fs,1/T)
  A=abs(np.fft.fft(xn.T)).T
  f=f[:int(N/2+1)]
  A=A[:int(N/2+1)]/N
  A[1:]=2*A[1:]
  A=abs(A)

  return f, A

def filtering(v, fs, Wn, ftype):
    filter_order=10; #filter order
    
    Fn=fs/2;

    [z,p,k] = signal.butter(filter_order,Wn/Fn,btype=ftype,output='zpk')

    sos = signal.zpk2sos(z,p,k); #g is same as k
    sos[0,0:3]=sos[0,0:3]/k 
    
    v_filter = signal.sosfilt(sos,v)*k
    return v_filter

def envelope_spectrum_band(x, fs, Wn):
    x_f = filtering(x, fs, Wn, 'band')
    env = np.abs(hilbert(x_f))
    f, A = fft_simple(env - np.mean(env), fs)
    return f, A, env, x_f

# --- helpers ---
def get_fault_frequency(config):
    Pd = (config['d1'] + config['d2'])/2
    unit = {
        'BPFI': 0.5*config['Nb']*(1 + config['Bd']/Pd*np.cos(config['a'])),
        'BPFO': 0.5*config['Nb']*(1 - config['Bd']/Pd*np.cos(config['a'])),
        'BSF' : (Pd/config['Bd'])*(1 - (config['Bd']/Pd*np.cos(config['a']))**2),
        'FTF' : (0.5*config['Nb']*(1 - config['Bd']/Pd*np.cos(config['a'])))/config['Nb']
    }
    label_map = {'I':'BPFI', 'O':'BPFO', 'B':'BSF'}
    fr = config['speed_rpm']/60
    FCF = unit[label_map[config['condition']]]
    FTF = unit['FTF']
    fault_freq = FCF*fr
    return fault_freq, FTF, FCF, unit

def generate_unit_impulse_response(t, beta_range, fn_range, rng):
    beta = rng.uniform(beta_range[0], beta_range[1])
    fn = rng.uniform(fn_range[0], fn_range[1])
    unit_fault = np.exp(-beta*t) * np.sin(2*np.pi*fn*t)
    return unit_fault, beta, fn

def fault_repeat_uncertainty(Unit_Fault, FCF, fr, t, Fs, FTF, condition, fr_std_ratio=0.02, slip=0.01, slip_std=0.001):
    """
    Minimal extension to repeat unit impacts with optional per-impact slip jitter.
    slip, slip_std are fractional jitter applied to the nominal step in samples.
    Defaults leave behavior unchanged.
    """
    # fr = fr + fr_std_ratio*fr*np.random.randn()
    Tp = 1.0/(FCF*fr)
    x = np.zeros_like(t)
    i = np.random.uniform(0, int(Tp*Fs))
    phi = np.random.uniform(0, 2*np.pi)
    if condition == 'O':
        Mod = np.ones_like(t)
    elif condition == 'I':
        Mod = 0.5*(1+np.cos(2*np.pi*fr*t + phi))
    else:
        Mod = (3+np.cos(2*np.pi*FTF*fr*t + phi))/8
    base_step = max(1, int(Tp*Fs))
    while i < len(t):
        idx = int(i)
        m = Mod[idx]
        rem = len(t) - idx
        if rem > 0:
            x[idx:] += m * Unit_Fault[:rem]

        # add small per-impact jitter in samples (fractional slip)
        if slip != 0.0 or slip_std != 0.0:
            # sample jitter drawn per-event (fractional), convert to samples
            frac = slip + slip_std * np.random.randn()
            jitter = int(np.round(frac * base_step))
        else:
            jitter = 0

        i += base_step + jitter
    return x

def real_cepstrum(x, n=None):

    spectrum = np.fft.fft(x, n=n)
    # ceps = np.fft.ifft(np.log(spectrum)).real
    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real


    return ceps

def def_features(v, fs, fault_freq, nlevel, f_range = 10, t_range = 0.001):
    t = np.arange(len(v)) / fs
    f, A = fft_simple(v-np.mean(v), fs)
    
    idx_f_range = np.where((f >= fault_freq-f_range) & (f <= fault_freq+f_range))[0]
    idx_t_range = np.where((t >= 1/fault_freq-t_range) & (t <= 1/fault_freq+t_range))[0]

    ###############################################################
    ######################## Basic statistics #####################
    ###############################################################
    rms = np.sqrt(np.mean(v**2))
    kurt = scipy.stats.kurtosis(v)
    
    ###############################################################
    ######################## Env Spectrum (raw) ###################
    ###############################################################
    env = np.abs(hilbert(v))
    f, A = fft_simple(env-np.mean(env), fs)
    env_raw = np.mean(A[idx_f_range])/np.sqrt(np.mean(A**2))

    ###############################################################
    ######################## Cepstrum ###################
    ###############################################################
    cep_signal = real_cepstrum(v-np.mean(v))
    cep = np.mean(np.abs(cep_signal[idx_t_range]))

    ###############################################################
    ######################## Kurtogram - maxK #####################
    ###############################################################
    _, _, _, _, maxK_, _, _, Wn = kg.fast_kurtogram(v, fs, nlevel=nlevel)
    
    
    ###############################################################
    ######################## Filtered Signal #####################
    ###############################################################
    f, A, env, x_filtered = envelope_spectrum_band(v, fs, Wn)
    env_filtered = np.mean(A[idx_t_range])/np.sqrt(np.mean(A**2))
    cep_signal = real_cepstrum(x_filtered-np.mean(x_filtered))
    cep_filtered = np.mean(np.abs(cep_signal[idx_t_range]))
    
    ###############################################################
    ######################## Final features #####################
    feat = [rms, kurt, env_raw, cep, maxK_, env_filtered, cep_filtered]
    
    feat_names = ['rms', 'kurt', 'env_raw', 'cepstrum_raw', 'kurt_filtered', 'env_filtered', 'cepstrum_filtered']
    
    return feat, feat_names

def def_features_seg(seg, fs, fault_freq, nlevel, f_range = 10, t_range = 0.001):
    feat_all = []
    for v in seg:
        feat = def_features(v, fs, fault_freq, nlevel, f_range, t_range)
        feat_all.append(feat)
    return np.array(feat_all)