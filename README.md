# EEG Artifact Subspace Reconstruction (ASR) & Bandpower Analysis

This project provides an interactive EEG signal processing platform using **Artifact Subspace Reconstruction (ASR)** and **bandpower computation**. It is designed to evaluate EEG quality through blink classification, intra/inter-subject calibration, and real-time analysis via a Streamlit web interface.

## 🚀 Features

- EEG preprocessing using ASR
- Bandpower feature extraction across different frequency bands
- Blink detection and classification
- Intra-subject and inter-subject generalization analysis
- Streamlit-based UI for interactive experimentation

## 🧠 What is ASR?

ASR (Artifact Subspace Reconstruction) is a method for automatically detecting and removing high-variance artifacts (e.g., eye blinks, muscle movements) from EEG data while preserving neural signals.

---

## 🛠️ Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/eeg-asr-streamlit.git
   cd eeg-asr-streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run main.py
   ```

---

## 📊 Input Format

Your EEG `.csv` files should be structured like this:

- Rows: Time samples
- Columns: EEG channels (e.g., `Ch_1`, `Ch_2`, ..., `Ch_24`)

Make sure sampling frequency (`fs`) is defined in your script or metadata.

---

## 🧪 Experiments Included

- **Bandpower Extraction**: Computes power in standard EEG frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- **Blink Classification**: Trains and tests a model to detect eye blinks
- **ASR Calibration**: Analyzes the effect of calibration on ASR performance
- **Generalization**: Tests models across subjects and sessions

---

## ✅ Requirements

All Python dependencies are listed in `requirements.txt`, including:

- `numpy`, `scipy`, `mne`, `pandas`, `matplotlib`
- `scikit-learn` for ML models
- `streamlit` for the UI

---

## 📌 Notes

- If you encounter `AttributeError: module 'scipy.integrate' has no attribute 'simps'`, make sure you have `scipy==1.10.1` installed.
- Designed to run locally, can be extended for cloud use or Raspberry Pi deployment.

---

## 📬 Contact

Feel free to reach out for questions, contributions, or collaborations!

- Author: Sumit Nayan
- Email: n.sumit@iitg.ac.in

---

## 📄 License

MIT License. See `LICENSE` file for details.
