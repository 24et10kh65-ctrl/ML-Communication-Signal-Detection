import h5py
import joblib
import matplotlib.style
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os


st.markdown("""
<style>
/* Main headings */
h1, h2, h3, h4, h5, h6 {
    color: #17365d !important;
}

/* Paragraphs and normal text */
.stMarkdown p,
.stMarkdown li,
.stMarkdown span,
label,
[data-testid="stWidgetLabel"] {
    color: #17365d !important;
}

/* Sidebar text */
[data-testid="stSidebar"] {
    color: #17365d !important;
}

/* Input labels */
.stSelectbox label,
.stSlider label {
    color: #17365d !important;
}
</style>
""", unsafe_allow_html=True)


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

st.set_page_config(
    page_title="ML Signal Detection",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #f5f7fc;
    }

    [data-testid="stSidebar"] {
        background-color: #e8eef9;
    }

    h1, h2, h3 {
        color: #17365d;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #dce3ef;
        padding: 16px;
        border-radius: 12px;
    }

    .stButton > button {
        background-color: #2457c5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #173b8c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


SIGNAL_TYPES = ["BPSK", "QPSK", "BFSK"]
SNR_VALUES = [-5, 0, 5, 10, 15]


def generate_signal(signal_type, num_bits=100,
                    samples_per_symbol=20, snr_db=5):
    bits = np.random.randint(0, 2, num_bits)
    t = np.arange(samples_per_symbol)

    if signal_type == "BPSK":
        symbols = 1 - 2 * bits
        signal = np.repeat(symbols, samples_per_symbol)

    elif signal_type == "QPSK":
        qbits = np.random.randint(0, 2, (num_bits, 2))
        i = 1 - 2 * qbits[:, 0]
        q = 1 - 2 * qbits[:, 1]
        symbols = (i + 1j * q) / np.sqrt(2)
        signal = np.repeat(symbols, samples_per_symbol)

    else:
        f0, f1 = 0.05, 0.15
        signal = np.concatenate([
            np.exp(
                2j * np.pi *
                (f0 if bit == 0 else f1) * t
            )
            for bit in bits
        ])

    power = np.mean(np.abs(signal) ** 2)
    bits_per_symbol = 2 if signal_type == "QPSK" else 1

    noise_power = (
    power * samples_per_symbol
    / (bits_per_symbol * (10 ** (snr_db / 10)))
    )

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal))
        + 1j * np.random.randn(len(signal))
    )

    return signal + noise


def calculate_ber(signal_type, num_bits=1000,
                  samples_per_symbol=20, snr_db=5):

    bits = np.random.randint(0, 2, num_bits)
    t = np.arange(samples_per_symbol)

    if signal_type == "BPSK":
        symbols = 1 - 2 * bits
        signal = np.repeat(symbols, samples_per_symbol)

    elif signal_type == "QPSK":
        qbits = np.random.randint(0, 2, (num_bits, 2))
        bits = qbits.flatten()

        i = 1 - 2 * qbits[:, 0]
        q = 1 - 2 * qbits[:, 1]
        symbols = (i + 1j * q) / np.sqrt(2)
        signal = np.repeat(symbols, samples_per_symbol)

    elif signal_type == "BFSK":
        f0, f1 = 0.05, 0.15
        signal = np.concatenate([
            np.exp(2j * np.pi * (f0 if bit == 0 else f1) * t)
            for bit in bits
        ])

    else:
        raise ValueError("Unsupported signal type")

    power = np.mean(np.abs(signal) ** 2)
    bits_per_symbol = 2 if signal_type == "QPSK" else 1

    noise_power = (
    power * samples_per_symbol
    / (bits_per_symbol * (10 ** (snr_db / 10)))
    )

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal))
        + 1j * np.random.randn(len(signal))
    )

    received_signal = signal + noise
    received_symbols = received_signal.reshape(-1, samples_per_symbol)

    if signal_type == "BPSK":
        received_bits = (
            np.mean(np.real(received_symbols), axis=1) < 0
        ).astype(int)

    elif signal_type == "QPSK":
        received_symbols = np.mean(received_symbols, axis=1)
        received_bits = np.column_stack((
            np.real(received_symbols) < 0,
            np.imag(received_symbols) < 0
        )).astype(int).flatten()

    else:
        ref0 = np.exp(2j * np.pi * 0.05 * t)
        ref1 = np.exp(2j * np.pi * 0.15 * t)

        energy0 = np.abs(
            np.sum(received_symbols * np.conj(ref0), axis=1)
        )
        energy1 = np.abs(
            np.sum(received_symbols * np.conj(ref1), axis=1)
        )
        received_bits = (energy1 > energy0).astype(int)

    errors = np.sum(bits != received_bits)
    total_bits = len(bits)

    ber = errors / total_bits
    error_percentage = ber * 100

    return ber, error_percentage



def extract_features(signal):
    real_signal = np.real(signal)
    magnitude = np.abs(signal)

    return [
        np.mean(real_signal),
        np.std(real_signal),
        np.mean(magnitude ** 2),
        np.std(magnitude),
        np.mean(real_signal ** 2)
    ]


@st.cache_resource
def train_model():
    X, y = [], []

    for signal_type in SIGNAL_TYPES:
        for snr in SNR_VALUES:
            for _ in range(200):
                signal = generate_signal(
                    signal_type, snr_db=snr
                )
                X.append(extract_features(signal))
                y.append(signal_type)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)

    snr_accuracies = []

    
    
    ber_values = {
        signal_type: [] for signal_type in SIGNAL_TYPES
    }
    ber_percentages = {
        signal_type: [] for signal_type in SIGNAL_TYPES
    }

    for snr in SNR_VALUES:
        for signal_type in SIGNAL_TYPES:
            total_errors = 0
            total_bits = 0

            for _ in range(100):
                ber, percentage = calculate_ber(
                    signal_type, snr_db=snr
                )

                bits_count = 2000 if signal_type == "QPSK" else 1000
                total_errors += round(ber * bits_count)
                total_bits += bits_count

            overall_ber = total_errors / total_bits

            ber_values[signal_type].append(overall_ber)
            ber_percentages[signal_type].append(
                overall_ber * 100
            )



    for snr in SNR_VALUES:
        X_snr, y_snr = [], []

        for signal_type in SIGNAL_TYPES:
            for _ in range(100):
                signal = generate_signal(
                    signal_type, snr_db=snr
                )
                X_snr.append(extract_features(signal))
                y_snr.append(signal_type)

        snr_predictions = model.predict(np.array(X_snr))
        snr_accuracy = accuracy_score(
            y_snr, snr_predictions
        )
        snr_accuracies.append(snr_accuracy * 100)

    return (
    model, test_accuracy, y_test, predictions,
    snr_accuracies, ber_values, ber_percentages
)


with st.sidebar:
    st.title("📡 Project Info")
    st.divider()

    st.markdown("### Project Title")
    st.write(
        "ML-Based Communication Signal Detection "
        "for Defence Applications"
    )

    st.markdown("### Technologies")
    st.write("- Python")
    st.write("- Machine Learning")
    st.write("- Random Forest")
    st.write("- NumPy")
    st.write("- Streamlit")

    st.markdown("### Signal Types")
    st.write("• BPSK")
    st.write("• QPSK")
    st.write("• BFSK")

    st.markdown("### Project Modules")
    st.write("1. Signal Generation")
    st.write("2. Noise Addition")
    st.write("3. Feature Extraction")
    st.write("4. ML Classification")
    st.write("5. Performance Evaluation")

    st.divider()
    st.caption("ENTC Cornerstone Project")

st.title("📡 ML-Based Communication Signal Detection")
st.subheader("Defence Applications | Simulation Dashboard")

st.write(
    "This project uses a Random Forest classifier to identify "
    "BPSK, QPSK and BFSK signals under different noise conditions."
)

st.info(
    "This is a synthetic simulation prototype, "
    "not a validated real-world defence system."
)

with st.spinner("Training the machine learning model..."):
    model, test_accuracy, y_test, predictions, snr_accuracies, ber_values, ber_percentages = (
    train_model()
)

col1, col2, col3 = st.columns(3)

col1.metric("Model", "Random Forest")
col2.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")
col3.metric("Signal Classes", "3")

st.divider()
st.header("🔍 Test a New Signal")

col1, col2 = st.columns(2)

with col1:
    signal_type = st.selectbox(
        "Choose signal type",
        SIGNAL_TYPES
    )

with col2:
    snr = st.slider(
        "Signal-to-Noise Ratio (SNR in dB)",
        min_value=-5,
        max_value=15,
        value=5,
        step=5
    )

if st.button("Generate and Classify Signal", type="primary"):
    signal = generate_signal(signal_type, snr_db=snr)
    features = np.array([extract_features(signal)])
    prediction = model.predict(features)[0]

    st.subheader("Classification Result")

    if prediction == signal_type:
        st.success(f"Predicted: {prediction} | Correct classification")
    else:
        st.warning(
            f"Predicted: {prediction} | "
            f"Generated signal: {signal_type}"
        )

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.real(signal[:400]))
    ax.set_title(f"Signal Waveform: {signal_type}, SNR = {snr} dB")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.header("📊 Model Evaluation")

tab1, tab2 = st.tabs(["Confusion Matrix", "SNR Performance"])

with tab1:
    cm = confusion_matrix(
        y_test, predictions, labels=SIGNAL_TYPES
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=SIGNAL_TYPES
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Communication Signal Classification")
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        SNR_VALUES,
        snr_accuracies,
        marker="o"
    )
    ax.set_title("Classification Accuracy at Different SNR Levels")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.write("SNR Performance Results")
    
    report_df = {
        "SNR (dB)": SNR_VALUES,
        "Accuracy (%)": [
            round(value, 2) for value in snr_accuracies
        ]
    }

    import pandas as pd

    report_df = pd.DataFrame(report_df)

    st.download_button(
        label="📥 Download SNR Report (CSV)",
        data=report_df.to_csv(index=False),
        file_name="snr_performance_report.csv",
        mime="text/csv"
    )

    st.dataframe({
        "SNR (dB)": SNR_VALUES,
        "Accuracy (%)": [
            round(value, 2) for value in snr_accuracies
        ]
    }, use_container_width=True)

    
    st.subheader("BER Performance at Different SNR Levels")

    fig, ax = plt.subplots(figsize=(8, 4))
    
    styles = {
        "BPSK": {"color": "blue", "marker": "o", "linestyle": "-"},
        "QPSK": {"color": "orange", "marker": "s", "linestyle": "--"},
        "BFSK": {"color": "green", "marker": "^", "linestyle": "-."}
    }

    for signal_type in SIGNAL_TYPES:
        ax.plot(
            SNR_VALUES,
            ber_values[signal_type],
            label=signal_type,
            linewidth=2,
            markersize=9,
            **styles[signal_type]
        )

    ax.legend()

    ax.set_title("BER Performance at Different SNR Levels")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Bit Error Percentage at Different SNR Levels")

    fig, ax = plt.subplots(figsize=(8, 4))
    
    styles = {
        "BPSK": {"color": "blue", "marker": "o", "linestyle": "-"},
        "QPSK": {"color": "orange", "marker": "s", "linestyle": "--"},
        "BFSK": {"color": "green", "marker": "^", "linestyle": "-."}
    }

    for signal_type in SIGNAL_TYPES:
        ax.plot(
            SNR_VALUES,
            ber_percentages[signal_type],
            label=signal_type,
            linewidth=2,
            markersize=9,
            **styles[signal_type]
        )

    ax.legend()

    ax.set_title("Bit Error Percentage at Different SNR Levels")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Percentage (%)")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Signal Feature Scatter Plot")

scatter_data = []

for signal_type in SIGNAL_TYPES:
    for _ in range(50):
        signal = generate_signal(signal_type, snr_db=5)
        features = extract_features(signal)

        scatter_data.append({
            "Signal Type": signal_type,
            "Mean Power": features[2],
            "Standard Deviation": features[1]
        })

import pandas as pd

scatter_df = pd.DataFrame(scatter_data)

fig, ax = plt.subplots(figsize=(8, 5))

for signal_type in SIGNAL_TYPES:
    data = scatter_df[
        scatter_df["Signal Type"] == signal_type
    ]

    ax.scatter(
        data["Mean Power"],
        data["Standard Deviation"],
        label=signal_type,
        alpha=0.7
    )

ax.set_title("Signal Feature Scatter Plot")
ax.set_xlabel("Mean Signal Power")
ax.set_ylabel("Standard Deviation")
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

st.subheader("Bandwidth Requirement")

bandwidth_data = {
    "BPSK": 1.0,
    "QPSK": 1.0,
    "BFSK": 2.0
}

bandwidth_df = pd.DataFrame({
    "Signal Type": list(bandwidth_data.keys()),
    "Normalized Bandwidth": list(bandwidth_data.values())
})

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    bandwidth_df["Signal Type"],
    bandwidth_df["Normalized Bandwidth"]
)

ax.set_title("Bandwidth Requirement of Different Modulation Schemes")
ax.set_xlabel("Signal Type")
ax.set_ylabel("Normalized Bandwidth")
ax.grid(axis="y", alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Bandwidth is shown in normalized units for comparison. "
    "Actual bandwidth in Hz depends on the symbol rate and sampling rate."
)


st.caption(
    "ENTC Cornerstone Project | ML Simulation Prototype"
)

# =====================================
# ANOMALY DETECTION
# =====================================

import streamlit as st
import numpy as np
from anomaly_detection import (
    train_anomaly_detector,
    analyze_signal,
)


@st.cache_resource
def get_anomaly_detector():
    rng = np.random.default_rng(42)

    reference_signals = [
        rng.normal(0, 1, 200)
        for _ in range(100)
    ]

    return train_anomaly_detector(reference_signals)


st.divider()
st.header("Anomaly Detection")
st.write(
    "Check whether a test signal looks unusual "
    "compared with the reference patterns."
)

st.warning(
    "Demo mode: uses synthetic test data, not real "
    "recorded communication signals."
)

st.subheader("Recorded Research Signal")
uploaded_file = st.file_uploader(
    "Upload a CSV file containing signal samples",
    type=["csv"],
    key="recorded_signal_csv",
)

if uploaded_file is not None:
    try:
        recorded_data = np.genfromtxt(
            uploaded_file,
            delimiter=",",
            dtype=float,
            comments="#",
        )

        recorded_data = np.asarray(recorded_data)
        recorded_data = np.squeeze(recorded_data)

        if recorded_data.ndim == 1:
            recorded_data = recorded_data[np.isfinite(recorded_data)]
            recorded_signal = recorded_data
        elif recorded_data.ndim == 2 and recorded_data.shape[1] >= 2:
            recorded_data = recorded_data[:, :2]
            recorded_data = recorded_data[
                np.all(np.isfinite(recorded_data), axis=1)
            ]
            recorded_signal = (
                recorded_data[:, 0] + 1j * recorded_data[:, 1]
            )
        else:
            recorded_signal = np.array([])

        if len(recorded_signal) < 10:
            st.error(
                "CSV mein kam se kam 10 valid samples hone chahiye."
            )
        else:
            st.success(
                f"Loaded {len(recorded_signal)} signal samples."
            )
            st.line_chart(np.abs(recorded_signal))

            if st.button(
                "Analyze Recorded Signal",
                key="analyze_recorded_signal_btn",
            ):
                detector = get_anomaly_detector()
                result = analyze_signal(
                    detector, recorded_signal
                )
                st.write("Status:", result["status"])
                st.write(
                    "Anomaly score:",
                    round(result["anomaly_score"], 5),
                )

    except Exception as error:
        st.error(f"Could not read the CSV: {error}")


if st.button("Analyze Signal for Anomalies",
             key="anomaly_detection_btn"):
    try:
        detector = get_anomaly_detector()
        rng = np.random.default_rng()

        test_signal = rng.normal(0, 1, 200)
        result = analyze_signal(detector, test_signal)

        st.write("Status:", result["status"])
        st.write(
            "Anomaly score:",
            round(result["anomaly_score"], 5),
        )

    except Exception as error:
        st.error(f"Analysis failed: {error}")

        # ==========================================
# RECORDED IQ SIGNAL ANALYSIS
# ==========================================

import h5py
import os

st.divider()
st.header("Recorded IQ Signal Analysis")

st.write(
    "Explore recorded IQ samples from the research dataset."
)

train_file = "dataset/subset_train.h5"

if os.path.exists(train_file):

    try:
        with h5py.File(train_file, "r") as data:
            total_samples = data["X"].shape[0]

            sample_index = st.number_input(
                "Select recorded sample index",
                min_value=0,
                max_value=total_samples - 1,
                value=0,
                step=1,
                key="recorded_iq_sample_index",
            )

            index = int(sample_index)

            iq_sample = data["X"][index]
            modulation_label = int(data["y_mod"][index])
            channel_label = int(data["y_chan"][index])
            snr_value = int(data["y_snr"][index])

        st.subheader(f"Recorded Sample #{index}")

        st.write("Modulation label:", modulation_label)

        st.write(
            "Channel condition:",
            "Clean" if channel_label == 0 else "Multipath",
        )

        st.write("SNR:", f"{snr_value} dB")

        st.caption(
            "Modulation is shown as its numeric dataset label. "
            "The label-to-modulation mapping must be verified "
            "from the dataset documentation."
        )

        st.subheader("I/Q Waveform")

        st.line_chart({
            "I (In-phase)": iq_sample[:, 0],
            "Q (Quadrature)": iq_sample[:, 1],
        })

    except Exception as error:
        st.error(f"Could not load recorded sample: {error}")

else:

    st.info(
        "Recorded IQ dataset is not included in the deployed demo "
        "because the dataset file is large. "
        "The Recorded IQ analysis is available in the local project version."
    )




    
# ==========================================
# REAL DATASET MODEL PREDICTION
# ==========================================

st.divider()
st.header("Real Dataset Model Prediction")

st.write(
    "Use the trained Machine Learning model "
    "to classify a recorded IQ signal."
)


def extract_model_features(signal):
    i = signal[:, 0].astype(np.float32)
    q = signal[:, 1].astype(np.float32)

    magnitude = np.sqrt(i**2 + q**2)
    power = i**2 + q**2

    return np.array([[
        np.mean(i),
        np.std(i),
        np.mean(q),
        np.std(q),
        np.mean(magnitude),
        np.std(magnitude),
        np.mean(power),
        np.std(power),
    ]], dtype=np.float32)


try:
    model = joblib.load("real_signal_model.joblib")

    with h5py.File("dataset/subset_test.h5", "r") as data:
        total_samples = data["X"].shape[0]

        
        available_labels = np.unique(data["y_mod"][:])

        selected_class = st.selectbox(
            "Choose actual class to test",
            options=[int(label) for label in available_labels],
            key="real_model_actual_class",
        )

        matching_indices = np.where(
            data["y_mod"][:] == selected_class
        )[0]

        index = int(matching_indices[0])
        signal = data["X"][index]
        actual_label = int(data["y_mod"][index])


        index = int(matching_indices[0])
        signal = data["X"][index]
        actual_label = int(data["y_mod"][index])

    if st.button("Classify Recorded Signal", key="real_model_predict"):
        features = extract_model_features(signal)
        predicted_label = int(model.predict(features)[0])

        st.write("Actual class:", actual_label)
        st.write("Predicted class:", predicted_label)

        if predicted_label == actual_label:
            st.success("Prediction matches the dataset label.")
        else:
            st.warning("Prediction differs from the dataset label.")

        st.subheader("Recorded I/Q Signal")
        st.line_chart({
            "I (In-phase)": signal[:, 0],
            "Q (Quadrature)": signal[:, 1],
        })

        st.caption(
            "Classes are numeric dataset labels. "
            "Their modulation names have not yet been verified."
        )

except FileNotFoundError:
    st.info(
        "Recorded IQ model or dataset is not included in the deployed demo "
        "because the required files are large. "
        "This feature is available in the local project version."
    )

except Exception as error:
    st.error(f"Prediction failed: {error}")


# ==========================================
# REAL MODEL PERFORMANCE EVALUATION
# ==========================================

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

st.divider()
st.header("Real Model Performance Evaluation")

st.write(
    "Evaluate the trained model on a larger set of "
    "recorded IQ test samples."
)

if st.button("Evaluate Model", key="evaluate_real_model"):
    try:

        if not os.path.exists("real_signal_model.joblib"):
            st.info(
                "Recorded IQ model is not available in the deployed demo."
            )
            st.stop()

        if not os.path.exists("dataset/subset_test.h5"):
            st.info(
                "Recorded IQ test dataset is not included in the deployed demo "
                "because the dataset file is large. "
                "The evaluation is available in the local project version."
            )
            st.stop()

        model = joblib.load("real_signal_model.joblib")

        with h5py.File("dataset/subset_test.h5", "r") as data:
            total = data["X"].shape[0]
            count = min(5000, total)

            rng = np.random.default_rng(42)
            indices = np.sort(
                rng.choice(total, size=count, replace=False)
            )

            signals = data["X"][indices]
            actual_labels = data["y_mod"][indices]

        features = extract_model_features(signals[0])

        # Extract features for all selected test signals
        all_features = []

        for signal in signals:
            all_features.append(extract_model_features(signal)[0])

        all_features = np.asarray(all_features, dtype=np.float32)
        predictions = model.predict(all_features)

        accuracy = accuracy_score(actual_labels, predictions)
        st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

        st.subheader("Confusion Matrix")

        labels = sorted(np.unique(actual_labels).tolist())
        matrix = confusion_matrix(
            actual_labels, predictions, labels=labels
        )

        st.dataframe(
            pd.DataFrame(
                matrix,
                index=[f"Actual {x}" for x in labels],
                columns=[f"Predicted {x}" for x in labels],
            )
        )

        st.subheader("Classification Report")

        report = classification_report(
            actual_labels,
            predictions,
            zero_division=0,
            output_dict=True,
        )

        st.dataframe(pd.DataFrame(report).transpose())

        st.caption(
            "Results apply to the selected test subset. "
            "Numeric labels are used because modulation names "
            "have not yet been verified."
        )

    except Exception as error:
        st.error(f"Evaluation failed: {error}")


