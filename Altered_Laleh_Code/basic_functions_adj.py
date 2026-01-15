import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Load & clean data
# ======================================================

def load_data(path):
    """
    Loads CSV data and standardizes column names.
    Expected input columns:
      - Patient_Anonmyized
      - Treatment_Day
      - TargetLesionLongDiam_mm
      - Study_Arm
    """

    df = pd.read_csv(path)

    df = df.rename(columns={
        "Patient_Anonmyized": "patient_id",
        "Treatment_Day": "day",
        "TargetLesionLongDiam_mm": "ld_mm",
        "Study_Arm": "arm"
    })

    df = df.dropna(subset=["patient_id", "day", "ld_mm", "arm"])
    df["day"] = df["day"].astype(float)
    df["ld_mm"] = pd.to_numeric(df["ld_mm"], errors="coerce")

    return df.sort_values(["patient_id", "day"])


# ======================================================
# Trend classification
# ======================================================

def classify_trend(ld_values, threshold=0.10):
    """
    Classifies tumor trajectory based on final vs baseline size.
    """
    baseline = ld_values[0]
    final = ld_values[-1]

    if baseline == 0:
        return "Fluctuate"

    pct_change = (final - baseline) / baseline

    if pct_change <= -threshold:
        return "Down"
    elif pct_change >= threshold:
        return "Up"
    else:
        return "Fluctuate"


# ======================================================
# RECIST-like response classification
# ======================================================

def classify_response(baseline, value):
    """
    RECIST-like response using percent change from baseline.
    """
    if baseline == 0:
        return "SD"

    pct_change = (value - baseline) / baseline

    if pct_change <= -0.30:
        return "PR"
    elif pct_change >= 0.20:
        return "PD"
    else:
        return "SD"


# ======================================================
# Plot single patient
# ======================================================

def plot_patient(df, patient_id):
    sub = df[df["patient_id"] == patient_id]

    if len(sub) < 2:
        print("Not enough data points to plot.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(sub["day"], sub["ld_mm"], marker="o", linewidth=3)

    plt.xlabel("Time (days)")
    plt.ylabel("Longest Diameter (mm)")
    plt.title(f"Patient {patient_id}")
    plt.grid(True)
    plt.show()


# ======================================================
# Spider plot
# ======================================================

def spider_plot(df, n_patients=40):
    plt.figure(figsize=(9, 7))

    colors = {
        "Up": "#d73027",
        "Down": "#1a9850",
        "Fluctuate": "#313695"
    }

    count = 0

    for pid in df["patient_id"].unique():
        sub = df[df["patient_id"] == pid]

        if len(sub) < 2:
            continue

        baseline = sub["ld_mm"].iloc[0]
        if baseline == 0:
            continue

        change = sub["ld_mm"] - baseline
        trend = classify_trend(sub["ld_mm"].values)

        plt.plot(
            sub["day"],
            change,
            color=colors[trend],
            alpha=0.7,
            linewidth=2
        )

        count += 1
        if count >= n_patients:
            break

    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Time (days)")
    plt.ylabel("Change from Baseline (mm)")
    plt.title("Spider Plot: Change in Target Lesion Over Time")

    handles = [
        plt.Line2D([0], [0], color=colors["Up"], lw=3, label="Up"),
        plt.Line2D([0], [0], color=colors["Down"], lw=3, label="Down"),
        plt.Line2D([0], [0], color=colors["Fluctuate"], lw=3, label="Fluctuate"),
    ]
    plt.legend(handles=handles)
    plt.grid(True)
    plt.show()


# ======================================================
# Arm-level trend summary
# ======================================================

def arm_trend_summary(df):
    records = []

    for arm in df["arm"].unique():
        sub_arm = df[df["arm"] == arm]

        for pid in sub_arm["patient_id"].unique():
            sub = sub_arm[sub_arm["patient_id"] == pid]

            if len(sub) < 2:
                continue

            trend = classify_trend(sub["ld_mm"].values)
            records.append({"arm": arm, "trend": trend})

    summary = pd.DataFrame(records)
    return summary.value_counts(["arm", "trend"]).unstack(fill_value=0)


# ======================================================
# Early prediction accuracy
# ======================================================

def early_prediction_accuracy(df, early_index=1, min_points=3):
    """
    Uses early_index-th measurement to predict final response.
    early_index = 1 → first follow-up
    """

    results = []

    for arm in df["arm"].unique():
        correct = 0
        total = 0

        sub_arm = df[df["arm"] == arm]

        for pid in sub_arm["patient_id"].unique():
            sub = sub_arm[sub_arm["patient_id"] == pid]

            if len(sub) < min_points or early_index >= len(sub):
                continue

            baseline = sub["ld_mm"].iloc[0]
            early = sub["ld_mm"].iloc[early_index]
            final = sub["ld_mm"].iloc[-1]

            pred = classify_response(baseline, early)
            true = classify_response(baseline, final)

            total += 1
            if pred == true:
                correct += 1

        acc = np.nan if total == 0 else round(100 * correct / total, 1)
        results.append({
            "arm": arm,
            "accuracy_%": acc,
            "n_patients": total
        })

    return pd.DataFrame(results)


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    # Load data
    df = load_data("real_data.csv")

    # Plot an example patient
    plot_patient(df, df["patient_id"].iloc[0])

    # Spider plot
    spider_plot(df, n_patients=40)

    # Arm-level trend summary
    print("\nArm-level trend summary:")
    print(arm_trend_summary(df))

    # Early prediction accuracy
    print("\nEarly prediction accuracy (first follow-up):")
    print(early_prediction_accuracy(df, early_index=1))

    print("\nEarly prediction accuracy (second follow-up):")
    print(early_prediction_accuracy(df, early_index=2))
