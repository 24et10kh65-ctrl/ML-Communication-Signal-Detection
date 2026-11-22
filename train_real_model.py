
import h5py
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
TRAIN_SAMPLES = 20000
TEST_SAMPLES = 5000


def extract_features(iq_samples):
    """Extract numerical features from I/Q signals."""
    features = []

    for signal in iq_samples:
        i = signal[:, 0].astype(np.float32)
        q = signal[:, 1].astype(np.float32)

        magnitude = np.sqrt(i**2 + q**2)
        power = i**2 + q**2

        features.append([
            np.mean(i),
            np.std(i),
            np.mean(q),
            np.std(q),
            np.mean(magnitude),
            np.std(magnitude),
            np.mean(power),
            np.std(power),
        ])

    return np.asarray(features, dtype=np.float32)


def load_random_subset(file_path, sample_count, rng):
    """Load a random subset without loading the full dataset."""
    with h5py.File(file_path, "r") as data:
        total = data["X"].shape[0]
        count = min(sample_count, total)

        indices = np.sort(
            rng.choice(total, size=count, replace=False)
        )

        signals = data["X"][indices]
        labels = data["y_mod"][indices]

    return signals, labels


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    print("Loading training samples...")
    X_train_raw, y_train = load_random_subset(
        "dataset/subset_train.h5", TRAIN_SAMPLES, rng
    )

    print("Loading test samples...")
    X_test_raw, y_test = load_random_subset(
        "dataset/subset_test.h5", TEST_SAMPLES, rng
    )

    print("Extracting features...")
    X_train = extract_features(X_train_raw)
    X_test = extract_features(X_test_raw)

    del X_train_raw, X_test_raw

    print("Training Random Forest model...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    joblib.dump(model, "real_signal_model.joblib")
    print("\nModel saved as real_signal_model.joblib")


if __name__ == "__main__":
    main()
