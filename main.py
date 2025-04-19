import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal, linalg, integrate
from scipy.stats import zscore, linregress
from scipy.signal import find_peaks, welch
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("EEG ASR Experiment Platform")

# --- Helper Functions ---
def load_eeg_from_csv(file):
    df = pd.read_csv(file)
    return df.values.T, df.columns.tolist()  # shape: (channels, samples), channel names

def preprocess_eeg(data, fs, notch_freq=50):
    data = data - np.mean(data, axis=1, keepdims=True)
    data = zscore(data, axis=1)
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

def eye_blink_count(X, XClean, fs):
    half_idx = X.shape[1] // 2
    signalOriginal = X[0, half_idx:]
    signalASR = XClean[0, half_idx:]
    thres1 = 6 * np.mean(np.abs(signalOriginal))
    blink_indicesOriginal, _ = find_peaks(signalOriginal, height=thres1, distance=0.5*fs)
    blink_indicesASR, _ = find_peaks(signalASR, height=thres1, distance=0.5*fs)
    return len(blink_indicesOriginal), len(blink_indicesASR)

def compute_rrmse(original, cleaned):
    min_len = min(original.shape[-1], cleaned.shape[-1])
    orig = original[..., :min_len]
    cln = cleaned[..., :min_len]
    return np.sqrt(np.mean((orig - cln) ** 2)) / np.sqrt(np.mean(orig ** 2))

def compute_bandpower(data, fs, band, window_sec=4):
    band = np.asarray(band)
    win = int(window_sec * fs)
    freqs, psd = signal.welch(data, fs, nperseg=win)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bp_abs = integrate.simps(psd[idx_band], dx=freqs[1] - freqs[0])
    bp_rel = bp_abs / integrate.simps(psd, dx=freqs[1] - freqs[0])
    return bp_abs, bp_rel

def detect_blinks(signal_data, fs):
    threshold = 6 * np.mean(np.abs(signal_data))
    peaks, _ = find_peaks(signal_data, height=threshold, distance=0.5*fs)
    return peaks

# --- UI: Data Upload ---
st.sidebar.header("Data Upload & Settings")
uploaded_files = st.sidebar.file_uploader("Upload EEG CSV files for subjects", type=["csv"], accept_multiple_files=True)
fs = st.sidebar.number_input("Sampling Rate (Hz)", value=500, step=1)
if not uploaded_files:
    st.info("Please upload at least one EEG CSV file to begin.")
    st.stop()

subject_names = [f.name for f in uploaded_files]
subject_data = {}
channel_names = None
for f in uploaded_files:
    data, ch_names = load_eeg_from_csv(f)
    subject_data[f.name] = data
    if channel_names is None:
        channel_names = ch_names

n_channels = len(channel_names)
bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 100)}

# --- UI: Experiment Selection ---
st.sidebar.header("Experiment")
experiment = st.sidebar.selectbox("Choose experiment", [
    "Intra-Subject Calibration Dependency",
    "Eyeblink Detection Type Classification",
    "Standardized Cleaning (Fixed Calibration)",
    "Inter-Subject Generalization (LOSO)",
    "Generalization Trend Analysis"
])

# --- Main Logic ---
if experiment == "Intra-Subject Calibration Dependency":
    st.header("Experiment 1.1: Calibration Length vs Eyeblink Cleaning Efficiency")
    subject = st.selectbox("Select Subject", subject_names)
    data = subject_data[subject]
    data = np.array([preprocess_eeg(ch[np.newaxis, :], fs).flatten() for ch in data])
    max_calib = int(data.shape[1] / fs) - 5
    calib_times = st.multiselect("Calibration Lengths (seconds)", list(range(10, max_calib, 10)), default=[10, 20, 30, 40, 50])
    results = []
    for calib_time in calib_times:
        calib_data = data[:, :calib_time*fs]
        state = asr_calibrate(calib_data, fs)
        padded = np.hstack([np.zeros((data.shape[0], int(fs / 4))), data[:, calib_time*fs:]])
        cleaned = asr_process(padded, fs, state)
        nblinks_o, nblinks_c = eye_blink_count(data, cleaned, fs)
        results.append((calib_time, nblinks_o, nblinks_c))
    df = pd.DataFrame(results, columns=["Calibration Time (s)", "Blinks Before", "Blinks After"])
    st.dataframe(df)
    fig, ax = plt.subplots()
    ax.plot(df["Calibration Time (s)"], df["Blinks Before"]-df["Blinks After"], marker='o')
    ax.set_xlabel("Calibration Time (s)")
    ax.set_ylabel("Number of Eyeblinks Removed")
    ax.set_title(f"Eyeblink Removal vs Calibration Length ({subject})")
    st.pyplot(fig)

elif experiment == "Eyeblink Detection Type Classification":
    st.header("Experiment 1.2: Eyeblink Detection Type Classification")
    subject = st.selectbox("Select Subject", subject_names)
    data = subject_data[subject]
    data = np.array([preprocess_eeg(ch[np.newaxis, :], fs).flatten() for ch in data])
    ch_idx = 0  # Use Ch_1 for blink detection
    max_calib = int(data.shape[1] / fs) - 5
    calib_times = list(range(10, max_calib, 10))
    blink_time_matrix = []
    cleaned_signals = []
    for calib_time in calib_times:
        calib_data = data[:, :calib_time*fs]
        state = asr_calibrate(calib_data, fs)
        padded = np.hstack([np.zeros((data.shape[0], int(fs / 4))), data[:, calib_time*fs:]])
        cleaned = asr_process(padded, fs, state)
        cleaned_signal = cleaned[ch_idx]
        cleaned_signals.append(cleaned_signal)
        blink_indices = detect_blinks(cleaned_signal, fs)
        blink_time_matrix.append(blink_indices / fs)
    raw_blink_indices = detect_blinks(data[ch_idx], fs)
    raw_blink_times = raw_blink_indices / fs
    blink_status = []
    tolerance = 0.2
    for blink_time in raw_blink_times:
        detected_at = []
        for blink_times in blink_time_matrix:
            detected = np.any(np.abs(blink_times - blink_time) < tolerance)
            detected_at.append(detected)
        blink_status.append(detected_at)
    blink_status = np.array(blink_status)
    type_I = [raw_blink_indices[i] for i, status in enumerate(blink_status) if np.all(status)]
    type_II = [raw_blink_indices[i] for i, status in enumerate(blink_status) if np.any(status) and not np.all(status)]
    type_III = [raw_blink_indices[i] for i, status in enumerate(blink_status) if not np.any(status)]
    st.write(f"Type I (detected at all lengths): {len(type_I)}")
    st.write(f"Type II (detected only at longer lengths): {len(type_II)}")
    st.write(f"Type III (missed at all lengths): {len(type_III)}")
    # Visualize one example from each type
    def plot_blink(signal_data, blink_idx, fs, title=""):
        window = int(1.5 * fs)
        start = max(blink_idx - window//2, 0)
        end = min(blink_idx + window//2, len(signal_data))
        t = np.arange(start, end) / fs
        fig, ax = plt.subplots()
        ax.plot(t, signal_data[start:end])
        ax.axvline(blink_idx/fs, color='red', linestyle='--', label='Blink')
        ax.set_title(title)
        ax.legend()
        st.pyplot(fig)
    if type_I:
        plot_blink(data[ch_idx], type_I[0], fs, "Type I Blink (Raw)")
    if type_II:
        plot_blink(data[ch_idx], type_II[0], fs, "Type II Blink (Raw)")
    if type_III:
        plot_blink(data[ch_idx], type_III[0], fs, "Type III Blink (Raw)")

elif experiment == "Standardized Cleaning (Fixed Calibration)":
    st.header("Experiment 1.3: Standardized Cleaning with Fixed Calibration")
    subject = st.selectbox("Select Subject", subject_names)
    data = subject_data[subject]
    data = np.array([preprocess_eeg(ch[np.newaxis, :], fs).flatten() for ch in data])
    calib_time = st.slider("Calibration Time (seconds)", min_value=20, max_value=60, value=30)
    calib_data = data[:, :calib_time*fs]
    state = asr_calibrate(calib_data, fs)
    padded = np.hstack([np.zeros((data.shape[0], int(fs / 4))), data[:, calib_time*fs:]])
    cleaned = asr_process(padded, fs, state)
    nblinks_o, nblinks_c = eye_blink_count(data, cleaned, fs)
    st.write(f"Number of eye blinks before ASR (raw): {nblinks_o}")
    st.write(f"Number of eye blinks after ASR (cleaned): {nblinks_c}")
    # RRMSE per channel
    rrmse_per_channel = []
    for i in range(data.shape[0]):
        min_len = min(data[i, calib_time*fs:].shape[0], cleaned[i].shape[0])
        orig = data[i, calib_time*fs:][:min_len]
        cln = cleaned[i, :min_len]
        rrmse = compute_rrmse(orig, cln)
        rrmse_per_channel.append(rrmse)
    st.write("RRMSE per channel:", rrmse_per_channel)
    # Bandpower
    bandpower_metrics = []
    for ch_idx, ch_name in enumerate(channel_names):
        orig = data[ch_idx, calib_time*fs:]
        cln = cleaned[ch_idx]
        metrics = {'Channel': ch_name}
        for band_name, band_range in bands.items():
            abs_orig, rel_orig = compute_bandpower(orig, fs, band_range)
            abs_cln, rel_cln = compute_bandpower(cln, fs, band_range)
            metrics[f"{band_name}_abs_raw"] = abs_orig
            metrics[f"{band_name}_rel_raw"] = rel_orig
            metrics[f"{band_name}_abs_clean"] = abs_cln
            metrics[f"{band_name}_rel_clean"] = rel_cln
        bandpower_metrics.append(metrics)
    st.dataframe(pd.DataFrame(bandpower_metrics).head())
    # Plot time domain
    seconds_to_plot = st.slider("Seconds to Plot", min_value=1, max_value=30, value=5)
    ch_idx = st.selectbox("Select Channel for Plot", range(n_channels), format_func=lambda x: channel_names[x])
    min_len = min(data[ch_idx, calib_time*fs:].shape[0], cleaned[ch_idx].shape[0], seconds_to_plot*fs)
    t = np.arange(min_len) / fs
    fig, ax = plt.subplots()
    ax.plot(t, data[ch_idx, calib_time*fs:][:min_len], label="Raw")
    ax.plot(t, cleaned[ch_idx, :min_len], label="Cleaned")
    ax.set_title(f"Time Domain Comparison ({channel_names[ch_idx]})")
    ax.legend()
    st.pyplot(fig)
    # Plot frequency domain
    f_raw, Pxx_raw = welch(data[ch_idx, calib_time*fs:], fs)
    f_clean, Pxx_clean = welch(cleaned[ch_idx], fs)
    fig, ax = plt.subplots()
    ax.semilogy(f_raw, Pxx_raw, label="Raw")
    ax.semilogy(f_clean, Pxx_clean, label="Cleaned")
    ax.set_title("Power Spectral Density")
    ax.legend()
    st.pyplot(fig)

elif experiment == "Inter-Subject Generalization (LOSO)":
    st.header("Experiment 2.1: Leave-One-Subject-Out Calibration Evaluation")
    st.write("Running LOSO cross-validation on all uploaded subjects...")
    summary = []
    bandpower_all = []
    for test_subject in subject_names:
        # Prepare calibration data from other subjects
        train_data = []
        for fname in subject_names:
            if fname == test_subject:
                continue
            d = subject_data[fname]
            d = np.array([preprocess_eeg(ch[np.newaxis, :], fs).flatten() for ch in d])
            train_data.append(d[:, :30*fs])
        calib_data = np.hstack(train_data)
        state = asr_calibrate(calib_data, fs)
        test_data = subject_data[test_subject]
        test_data = np.array([preprocess_eeg(ch[np.newaxis, :], fs).flatten() for ch in test_data])
        padded = np.hstack([np.zeros((test_data.shape[0], int(fs / 4))), test_data])
        cleaned = asr_process(padded, fs, state)
        min_len = min(test_data.shape[1], cleaned.shape[1])
        test_data = test_data[:, :min_len]
        cleaned = cleaned[:, :min_len]
        nblinks_o, nblinks_c = eye_blink_count(test_data, cleaned, fs)
        rrmse = np.mean([compute_rrmse(test_data[i], cleaned[i]) for i in range(test_data.shape[0])])
        band_changes = {}
        for band_name, band_range in bands.items():
            bp_raw = np.mean([compute_bandpower(test_data[i], fs, band_range)[0] for i in range(test_data.shape[0])])
            bp_clean = np.mean([compute_bandpower(cleaned[i], fs, band_range)[0] for i in range(cleaned.shape[0])])
            band_changes[f"{band_name}_change"] = bp_clean - bp_raw
        summary.append({
            "subject": test_subject,
            "blinks_raw": nblinks_o,
            "blinks_clean": nblinks_c,
            "avg_rrmse": rrmse,
            **band_changes
        })
        # Plot time/frequency for Ch_1
        st.subheader(f"Subject {test_subject} - Ch_1")
        t = np.arange(min_len) / fs
        fig, ax = plt.subplots()
        ax.plot(t[:5*fs], test_data[0, :5*fs], label="Raw")
        ax.plot(t[:5*fs], cleaned[0, :5*fs], label="Cleaned")
        ax.set_title("Time Domain (first 5s)")
        ax.legend()
        st.pyplot(fig)
        f_raw, Pxx_raw = welch(test_data[0], fs)
        f_clean, Pxx_clean = welch(cleaned[0], fs)
        fig, ax = plt.subplots()
        ax.semilogy(f_raw, Pxx_raw, label="Raw")
        ax.semilogy(f_clean, Pxx_clean, label="Cleaned")
        ax.set_title("Power Spectral Density")
        ax.legend()
        st.pyplot(fig)
    df_summary = pd.DataFrame(summary)
    st.dataframe(df_summary)

elif experiment == "Generalization Trend Analysis":
    st.header("Experiment 2.2: Generalization Trend Analysis")
    # Assume previous LOSO summary
    st.write("Trend analysis across subjects (requires LOSO run above).")
    # For demo, run LOSO as above
    summary = []
    for test_subject in subject_names:
        train_data = []
        for fname in subject_names:
            if fname == test_subject:
                continue
            d = subject_data[fname]
            d = np.array([preprocess_eeg(ch[np.newaxis, :], fs).flatten() for ch in d])
            train_data.append(d[:, :30*fs])
        calib_data = np.hstack(train_data)
        state = asr_calibrate(calib_data, fs)
        test_data = subject_data[test_subject]
        test_data = np.array([preprocess_eeg(ch[np.newaxis, :], fs).flatten() for ch in test_data])
        padded = np.hstack([np.zeros((test_data.shape[0], int(fs / 4))), test_data])
        cleaned = asr_process(padded, fs, state)
        min_len = min(test_data.shape[1], cleaned.shape[1])
        test_data = test_data[:, :min_len]
        cleaned = cleaned[:, :min_len]
        nblinks_o, nblinks_c = eye_blink_count(test_data, cleaned, fs)
        rrmse = np.mean([compute_rrmse(test_data[i], cleaned[i]) for i in range(test_data.shape[0])])
        band_changes = {}
        for band_name, band_range in bands.items():
            bp_raw = np.mean([compute_bandpower(test_data[i], fs, band_range)[0] for i in range(test_data.shape[0])])
            bp_clean = np.mean([compute_bandpower(cleaned[i], fs, band_range)[0] for i in range(cleaned.shape[0])])
            band_changes[f"{band_name}_change"] = bp_clean - bp_raw
        summary.append({
            "subject": test_subject,
            "blinks_raw": nblinks_o,
            "blinks_clean": nblinks_c,
            "avg_rrmse": rrmse,
            **band_changes
        })
    df = pd.DataFrame(summary)
    df = df.sort_values('avg_rrmse').reset_index(drop=True)
    x = np.arange(len(df))
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(x, df['avg_rrmse'], 'o-', color='tab:blue')
    axs[0].set_title('RRMSE Trend Across Subjects')
    axs[0].set_ylabel('RRMSE')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(df.subject, rotation=45)
    for band, color in zip(['Alpha', 'Beta', 'Gamma'], ['tab:red', 'tab:green', 'tab:orange']):
        axs[1].plot(x, df[f'{band}_change'], 'o--', label=f'{band} Band', color=color)
    axs[1].set_title('Band Power Change Trends')
    axs[1].set_ylabel('Power Change (Clean - Raw)')
    axs[1].legend()
    blink_reduction = df['blinks_raw'] - df['blinks_clean']
    axs[2].bar(x, blink_reduction, color='tab:purple', alpha=0.7)
    axs[2].set_title('Eye Blink Reduction Trend')
    axs[2].set_ylabel('Blinks Removed')
    axs[2].set_xlabel('Subject (Ordered)')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(df.subject, rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    # Statistical trend analysis
    def analyze_trend(y):
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend': 'Increasing' if slope > 0 else 'Decreasing',
            'significance': 'Significant' if p_value < 0.05 else 'Not Significant'
        }
    st.write("Trend Analysis Results:")
    st.json({
        "RRMSE": analyze_trend(df['avg_rrmse']),
        "Alpha Power": analyze_trend(df['Alpha_change']),
        "Beta Power": analyze_trend(df['Beta_change']),
        "Gamma Power": analyze_trend(df['Gamma_change']),
        "Blink Reduction": analyze_trend(blink_reduction)
    })

st.sidebar.markdown("---")
st.sidebar.info("Developed for EEG ASR experiment analysis. Upload subject CSVs and explore all experiments interactively.")
