
import numpy as np
from sklearn.ensemble import IsolationForest


def extract_signal_features(signal):
    """Extract basic statistical features from a signal."""
    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    magnitude = np.abs(signal)
    real_part = np.real(signal)

    return [
        float(np.mean(magnitude)),
        float(np.std(magnitude)),
        float(np.mean(magnitude ** 2)),
        float(np.std(real_part)),
    ]


def train_anomaly_detector(reference_signals):
    """
    Train on reference signals considered normal.
    Each item in reference_signals is a signal array.
    """
    if len(reference_signals) < 10:
        raise ValueError(
            "Provide at least 10 reference signals."
        )

    features = np.array([
        extract_signal_features(signal)
        for signal in reference_signals
    ])

    detector = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    detector.fit(features)
    return detector


def analyze_signal(detector, signal):
    """
    Compare a signal with the reference patterns.
    This flags unusual patterns; it does not prove interference.
    """
    features = np.array([
        extract_signal_features(signal)
    ])

    label = detector.predict(features)[0]
    score = float(detector.decision_function(features)[0])

    return {
        "status": (
            "Unusual pattern detected"
            if label == -1
            else "Within reference pattern"
        ),
        "anomaly_score": score,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Create synthetic reference signals for testing
    reference_signals = [
        rng.normal(0, 1, 200)
        for _ in range(100)
    ]

    detector = train_anomaly_detector(reference_signals)

    test_signal = rng.normal(0, 1, 200)
    result = analyze_signal(detector, test_signal)

    print("Test result:")
    print("Status:", result["status"])
    print("Anomaly score:", result["anomaly_score"])


# ==========================================
# ANOMALY DETECTION SECTION
# Synthetic demo now; recorded data later
# ==========================================

import numpy as np
import streamlit as st
from anomaly_detection import (
    train_anomaly_detector,
    analyze_signal,
)


def _anomaly_demo_signal(modulation, snr_db, rng):
    """Generate a synthetic communication signal for demonstration."""
    samples_per_symbol = 20
    num_symbols = 100
    bits = rng.integers(0, 2, num_symbols)

    if modulation == "BPSK":
        symbols = (2 * bits - 1).astype(complex)
        signal = np.repeat(symbols, samples_per_symbol)

    elif modulation == "QPSK":
        bits_i = rng.integers(0, 2, num_symbols)
        bits_q = rng.integers(0, 2, num_symbols)
        symbols = (
            (2 * bits_i - 1)
            + 1j * (2 * bits_q - 1)
        ) / np.sqrt(2)
        signal = np.repeat(symbols, samples_per_symbol)

    else:  # BFSK
        t = np.arange(samples_per_symbol)
        pieces = []

        for bit in bits:
            frequency = 0.05 if bit == 1 else -0.05
            piece = np.exp(1j * 2 * np.pi * frequency * t)
            pieces.extend(piece)

        signal = np.asarray(pieces, dtype=complex)

    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=len(signal))
        + 1j * rng.normal(size=len(signal))
    )

    return signal + noise


@st.cache_resource
def _get_anomaly_demo_detector():
    """Train on synthetic reference signals for the demo."""
    rng = np.random.default_rng(42)
    reference_signals = []

    for modulation in ["BPSK", "QPSK", "BFSK"]:
        for snr in [-5, 0, 5, 10, 15]:
            for _ in range(10):
                reference_signals.append(
                    _anomaly_demo_signal(modulation, snr, rng)
                )

    return train_anomaly_detector(reference_signals)


# This section appears below the existing dashboard.
st.divider()
st.header("🔎 Anomaly Detection")
st.write(
    "Compare a generated signal with reference signal patterns "
    "and flag statistically unusual samples."
)

st.info(
    "Demo mode: uses synthetic signals, not recorded RF data. "
    "An unusual result does not prove interference."
)

with st.container(border=True):
    anomaly_modulation = st.selectbox(
        "Select signal type",
        ["BPSK", "QPSK", "BFSK"],
        key="anomaly_modulation",
    )

    anomaly_snr = st.slider(
        "Signal-to-noise ratio (SNR in dB)",
        min_value=-5,
        max_value=15,
        value=5,
        step=5,
        key="anomaly_snr",
    )

    if st.button("Analyze Signal for Anomalies"):
        try:
            detector = _get_anomaly_demo_detector()
            rng = np.random.default_rng()

            test_signal = _anomaly_demo_signal(
                anomaly_modulation,
                anomaly_snr,
                rng,
            )

            result = analyze_signal(detector, test_signal)

            st.subheader("Analysis Result")
            st.write("**Status:**", result["status"])
            st.write(
                "**Anomaly score:**",
                round(result["anomaly_score"], 5),
            )

            st.caption(
                "The score is a model score, not a probability. "
                "Results need validation on representative recorded data."
            )

        except Exception as error:
            st.error(f"Anomaly analysis failed: {error}")
