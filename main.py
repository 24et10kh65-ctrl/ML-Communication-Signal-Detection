
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Module 1: Signal generation
def generate_signal(signal_type, num_bits=100, samples_per_symbol=20, snr_db=5):
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

    else:  # BFSK
        f0, f1 = 0.05, 0.15
        signal = np.concatenate([
            np.exp(2j * np.pi * (f0 if bit == 0 else f1) * t)
            for bit in bits
        ])

    # Add complex Gaussian noise
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )

    return signal + noise


# Module: BER Calculation

def calculate_ber(signal_type, num_bits=1000, samples_per_symbol=20, snr_db=5):

    # Generate transmitted bits
    bits = np.random.randint(0, 2, num_bits)
    t = np.arange(samples_per_symbol)

    # Generate modulated signal
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

    # Add Gaussian noise
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / (10 ** (snr_db / 10))

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal))
        + 1j * np.random.randn(len(signal))
    )

    received_signal = signal + noise

    # Demodulation
    received_symbols = received_signal.reshape(
        -1, samples_per_symbol
    )

    if signal_type == "BPSK":
        detected = np.mean(received_symbols.real, axis=1)
        received_bits = (detected < 0).astype(int)

    elif signal_type == "QPSK":
        detected = np.mean(received_symbols, axis=1)

        received_i = (detected.real < 0).astype(int)
        received_q = (detected.imag < 0).astype(int)

        received_bits = np.column_stack(
            (received_i, received_q)
        ).flatten()

    else:  # BFSK
        reference_0 = np.exp(2j * np.pi * 0.05 * t)
        reference_1 = np.exp(2j * np.pi * 0.15 * t)

        energy_0 = np.abs(
            np.sum(received_symbols * np.conj(reference_0), axis=1)
        ) ** 2

        energy_1 = np.abs(
            np.sum(received_symbols * np.conj(reference_1), axis=1)
        ) ** 2

        received_bits = (energy_1 > energy_0).astype(int)

    # Calculate BER
    bit_errors = np.sum(bits != received_bits)
    total_bits = len(bits)

    ber = bit_errors / total_bits
    percentage_error = ber * 100

    return bit_errors, total_bits, ber, percentage_error



# Module 2: Feature extraction
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


# Module 3: Create a larger labelled dataset
X = []
y = []

signal_types = ["BPSK", "QPSK", "BFSK"]


# Generate training data at multiple SNR levels
training_snr_values = [-5, 0, 5, 10, 15]

for signal_type in signal_types:
    for snr in training_snr_values:
        for _ in range(200):
            signal = generate_signal(signal_type, snr_db=snr)
            X.append(extract_features(signal))
            y.append(signal_type)


X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Samples per signal type:", len(X) // len(signal_types))


# Module 4: Split data and train the ML model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, random_state=42)
)

model.fit(X_train, y_train)


# Module 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Machine Learning Results ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Classification accuracy: {accuracy * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, labels=signal_types))

cm = confusion_matrix(y_test, y_pred, labels=signal_types)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=signal_types
)
display.plot(cmap="Blues")
plt.title("Communication Signal Classification")
plt.tight_layout()
plt.show()


# Module 6: Test a new simulated signal
new_signal = generate_signal("QPSK", snr_db=5)
new_features = np.array([extract_features(new_signal)])
prediction = model.predict(new_features)[0]

print("\n--- New Signal Prediction ---")
print("Actual signal type: QPSK")
print("Predicted signal type:", prediction)


# Module 5: Testing at Different SNR Levels

print("\n--- SNR Performance Testing ---")

snr_values = [-5, 0, 5, 10, 15]
snr_accuracies = []

for snr in snr_values:
    X_snr = []
    y_snr = []

    for signal_type in signal_types:
        for _ in range(100):
            signal = generate_signal(signal_type, snr_db=snr)
            X_snr.append(extract_features(signal))
            y_snr.append(signal_type)

    X_snr = np.array(X_snr)
    y_snr = np.array(y_snr)

    predictions = model.predict(X_snr)
    snr_accuracy = accuracy_score(y_snr, predictions)

    snr_accuracies.append(snr_accuracy * 100)

    print(f"SNR: {snr:>3} dB | Accuracy: {snr_accuracy * 100:.2f}%")


# Plot accuracy at different SNR levels
plt.figure(figsize=(8, 5))
plt.plot(snr_values, snr_accuracies, marker="o")
plt.title("ML Classification Accuracy at Different SNR Levels")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.show()


# Module 6: Save SNR Results for Project Report

import csv

results_file = "snr_results.csv"

with open(results_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["SNR (dB)", "Classification Accuracy (%)"])

    for snr, accuracy_value in zip(snr_values, snr_accuracies):
        writer.writerow([snr, round(accuracy_value, 2)])

print("\nSNR results saved to:", results_file)


# ==========================================
# MODULE: BIT ERROR RATE (BER) ANALYSIS
# ==========================================

print("\n--- BER PERFORMANCE ANALYSIS ---")

ber_snr_values = [-5, 0, 5, 10, 15]

ber_results = {}

for signal_type in signal_types:

    ber_values = []

    print(f"\nSignal Type: {signal_type}")

    for snr in ber_snr_values:

        errors = 0
        total = 0

        # Average over multiple transmissions
        for _ in range(20):

            bit_errors, total_bits, ber, error_percent = (
                calculate_ber(
                    signal_type,
                    num_bits=1000,
                    snr_db=snr
                )
            )

            errors += bit_errors
            total += total_bits

        overall_ber = errors / total
        overall_percentage = overall_ber * 100

        ber_values.append(overall_ber)

        print(
            f"SNR: {snr:>3} dB | "
            f"BER: {overall_ber:.5f} | "
            f"Percentage Error: {overall_percentage:.2f}%"
        )

    ber_results[signal_type] = ber_values


# Plot BER performance
plt.figure(figsize=(9, 5))

for signal_type, values in ber_results.items():
    plt.plot(
        ber_snr_values,
        values,
        marker="o",
        label=signal_type
    )

plt.title("BER Performance at Different SNR Levels")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Plot Percentage Error
plt.figure(figsize=(9, 5))

for signal_type, values in ber_results.items():
    percentages = [value * 100 for value in values]

    plt.plot(
        ber_snr_values,
        percentages,
        marker="o",
        label=signal_type
    )

plt.title("Bit Error Percentage at Different SNR Levels")
plt.xlabel("SNR (dB)")
plt.ylabel("Percentage Error (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

