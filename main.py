import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal, linalg
from scipy.stats import zscore
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

st.title("EEG Signal Processing with ASR")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your EEG CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
    fs = st.number_input("Sampling Rate (Hz)", value=500, step=1)
    n_channels = df.shape[1]
    channel_names = df.columns.tolist()

    # --- Calibration Time Selection ---
    max_calib = int(df.shape[0] / fs)
    calib_time = st.slider("Calibration Time (seconds)", min_value=5, max_value=max_calib, value=30)

    # --- Preprocessing and ASR Functions (from your code) ---
    def preprocess_eeg(data, fs, notch_freq=50):
        data = data - np.mean(data)
        data = zscore(data)
        b, a = signal.butter(4, [0.5, 100], btype='bandpass', fs=fs)
        data = signal.filtfilt(b, a, data)
        b_notch, a_notch = signal.iirnotch(notch_freq, 30, fs)
        return signal.filtfilt(b_notch, a_notch, data)

    def asr_calibrate(X, fs, cutoff=20):
        C, S = X.shape
        b, a = signal.butter(4, [1, 50], btype='bandpass', fs=fs)
        X_filtered = signal.filtfilt(b, a, X, axis=1)
        cov = X_filtered @ X_filtered.T / S
        M = linalg.sqrtm(cov).real
        D, V = np.linalg.eigh(M)
        order = np.argsort(D)[::-1]
        V = V[:, order]
        T = np.diag(np.mean(X_filtered, axis=1) + cutoff * np.std(X_filtered, axis=1)) @ V.T
        return {'M': M, 'T': T, 'b': b, 'a': a}

    def asr_process(data, fs, state, window_len=0.5, lookahead=0.25):
        C, S = data.shape
        N = int(window_len * fs)
        P = int(lookahead * fs)
        processed = np.zeros_like(data)
        for i in range(0, S, N):
            window = data[:, i:i+N]
            if window.shape[1] < N:
                break
            cov = window @ window.T / N
            D, V = np.linalg.eigh(cov)
            order = np.argsort(D)[::-1]
            V = V[:, order]
            D = D[order]
            keep = D < np.sum((state['T'] @ V)**2, axis=0)
            if np.sum(keep) < C - int(0.66 * C):
                keep[:int(C * (1 - 0.66))] = True
            if not np.all(keep):
                R = state['M'] @ np.linalg.pinv((V.T @ state['M']) * keep) @ V.T
            else:
                R = np.eye(C)
            processed[:, i:i+N] = R @ window
        return processed[:, P:]

    # --- Run Processing ---
    eeg_data = df.values.T  # shape: (channels, samples)
    eeg_data = np.array([preprocess_eeg(ch, fs) for ch in eeg_data])
    calib_data = eeg_data[:, :calib_time*fs]
    state = asr_calibrate(calib_data, fs)
    padded = np.hstack([np.zeros((eeg_data.shape[0], int(fs / 4))), eeg_data[:, calib_time*fs:]])
    cleaned = asr_process(padded, fs, state)

    # --- Channel and Time Selection for Plotting ---
    channel_idx = st.selectbox("Select Channel", range(n_channels), format_func=lambda x: channel_names[x])
    seconds_to_plot = st.slider("Seconds to Plot", min_value=1, max_value=120, value=10)
    n_samples_to_plot = seconds_to_plot * fs
    total_samples = cleaned.shape[1]
    if n_samples_to_plot > total_samples:
        n_samples_to_plot = total_samples
    time = np.arange(n_samples_to_plot) / fs
    raw_truncated = eeg_data[channel_idx, -n_samples_to_plot:]
    cleaned_truncated = cleaned[channel_idx, -n_samples_to_plot:]

    # --- Plotting ---
    fig, ax = plt.subplots(2, 1, figsize=(14, 6))
    ax[0].plot(time, raw_truncated, color="orange")
    ax[0].set_title(f"Raw EEG - {channel_names[channel_idx]}")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(True)

    ax[1].plot(time, cleaned_truncated, color="green")
    ax[1].set_title(f"Cleaned EEG (ASR) - {channel_names[channel_idx]}")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude")
    ax[1].grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Eye Blink Detection ---
    def eye_blink_count(X, XClean, fs):
        half_idx = X.shape[1] // 2
        signalOriginal = X[0, half_idx:]
        signalASR = XClean[0, half_idx:]
        thres1 = 6 * np.mean(np.abs(signalOriginal))
        blink_indicesOriginal, _ = find_peaks(signalOriginal, height=thres1, distance=0.5*fs)
        blink_indicesASR, _ = find_peaks(signalASR, height=thres1, distance=0.5*fs)
        return len(blink_indicesOriginal), len(blink_indicesASR)

    nblinks_o, nblinks_c = eye_blink_count(eeg_data, cleaned, fs)
    st.write(f"Number of eye blinks before ASR (raw): {nblinks_o}")
    st.write(f"Number of eye blinks after ASR (cleaned): {nblinks_c}")

    # --- Eye Blink Visualization ---
    def plot_eye_blinks(eeg_raw, eeg_cleaned, fs):
        half_idx = eeg_cleaned.shape[1] // 2
        signal_cleaned = eeg_cleaned[0, half_idx:]
        signal_raw = eeg_raw[0, -signal_cleaned.shape[0]:]
        time = np.arange(signal_cleaned.shape[0]) / fs
        threshold = 6 * np.mean(np.abs(signal_raw))
        blink_raw_idx, _ = find_peaks(signal_raw, height=threshold, distance=0.5 * fs)
        blink_cleaned_idx, _ = find_peaks(signal_cleaned, height=threshold, distance=0.5 * fs)
        fig, axs = plt.subplots(2, 1, figsize=(14, 6))
        axs[0].plot(time, signal_raw, label="Raw EEG (Ch_1)", color='blue')
        axs[0].plot(blink_raw_idx / fs, signal_raw[blink_raw_idx], 'rx', label="Eye Blinks")
        axs[0].set_title("Raw EEG Signal (Second Half)")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()
        axs[1].plot(time, signal_cleaned, label="Cleaned EEG (Ch_1)", color='green')
        axs[1].plot(blink_cleaned_idx / fs, signal_cleaned[blink_cleaned_idx], 'rx', label="Eye Blinks After Cleaning")
        axs[1].set_title("Cleaned EEG Signal (Second Half)")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend()
        plt.tight_layout()
        return fig

    st.pyplot(plot_eye_blinks(eeg_data, cleaned, fs))
else:
    st.info("Please upload a CSV file to begin.")


