import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")



# Read file, remove invalid raws and normalize data
DATA_PATH = "toolwindow_data.csv"
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["user_id","timestamp","event"])

df["user_id"] = df["user_id"].astype(str)
df["event"] = df["event"].astype(str).str.strip().str.lower()
df["open_type"] = df["open_type"].astype(str).str.strip().str.lower()
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df["timestamp"] = df["timestamp"].astype(np.int64)

df["_order"] = df["event"].map({"closed":0, "opened":1}).fillna(1)
df = df.sort_values(["user_id","timestamp","_order"]).drop(columns=["_order"])



# Get episodes
episodes = []

for id, frame in df.groupby("user_id", sort=False):
    current = None
    for _, row in frame.iterrows():
        event = row["event"]
        time = row["timestamp"]
        if event == "opened":
            typ = row.get("open_type")
            if pd.isna(typ) or typ not in ("manual", "auto"):
                continue
            if current is None:
                current = (time, typ)
            else:
                start, typ1 = current
                if time > start:
                    episodes.append((id, typ1, time - start))
                current = (time, typ)
        elif event == "closed":
            if current is not None:
                start, t = current
                if time > start:
                    episodes.append((id, t, time - start))
                current = None
            else:
                pass

episodes_table = pd.DataFrame(episodes, columns=["user_id","open_type", "duration_ms"])
episodes_table["duration_s"] = episodes_table["duration_ms"] / 1000.0



# Apply metrics

# Mann–Whitney U test
def mannwhitney(a: np.ndarray, b: np.ndarray):
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")

    return U, p


# Cliff’s Delta
def cliffs_delta(a: np.ndarray, b: np.ndarray):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    delta = (2 * U) / (n * m) - 1

    return delta


def cliffs_delta_ci(a, b):
    alpha = 0.05
    rng = np.random.default_rng(50)
    deltas = []
    for _ in range(3000):
        a1 = rng.choice(a, size=len(a), replace=True)
        b1 = rng.choice(b, size=len(b), replace=True)
        deltas.append(cliffs_delta(a1, b1))
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return lo, hi


# Median difference
def median_diff_ci(a, b):
    alpha = 0.05
    rng = np.random.default_rng(50)
    diffs = []
    for _ in range(3000):
        a1 = rng.choice(a, size=len(a), replace=True)
        b1 = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.median(a1) - np.median(b1))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return np.median(a) - np.median(b), lo, hi


def getSummary(episodes_t: pd.DataFrame):
    manual = episodes_t.loc[episodes_t["open_type"] == "manual", "duration_s"].values
    auto = episodes_t.loc[episodes_t["open_type"] == "auto", "duration_s"].values

    summary = {
        "manual": getStats(manual),
        "auto": getStats(auto),
    }

    U, p = mannwhitney(manual, auto)
    delta = cliffs_delta(manual, auto)
    delta_ci = cliffs_delta_ci(manual, auto)
    med_diff, lo, hi = median_diff_ci(manual, auto)

    results = {
        "mannwhitney_U": U,
        "p": p,
        "cliffs_delta": delta,
        "cliffs_delta_CI": delta_ci,
        "median_diff_s": med_diff,
        "median_diff_CI": (lo, hi),
        "summary_stats": summary,
    }
    return results


def getStats(durations_s: np.ndarray) -> dict:
    return {
        "count": int(len(durations_s)),
        "mean_s": float(np.mean(durations_s)),
        "median_s": float(np.median(durations_s)),
        "std_s": float(np.std(durations_s, ddof=1)),
        "p25_s": float(np.percentile(durations_s, 25)),
        "p75_s": float(np.percentile(durations_s, 75)),
        "max_s": float(np.max(durations_s)),
    }


def individualComprasion(episodes_tb: pd.DataFrame):
    per_user = (
        episodes_tb
        .groupby(["user_id", "open_type"])["duration_s"]
        .median()
        .reset_index()
        .pivot(index="user_id", columns="open_type", values="duration_s")
    )

    paired = per_user.dropna(subset=["manual", "auto"]).copy()

    paired["diff"] = paired["manual"] - paired["auto"]

    stat, p = stats.wilcoxon(paired["manual"], paired["auto"], alternative="two-sided")

    median_diff = np.median(paired["diff"])
    ci_low, ci_high = np.percentile(paired["diff"], [2.5, 97.5])

    results = {
        "users_paired": len(paired),
        "median_diff": float(median_diff),
        "median_diff_CI": (float(ci_low), float(ci_high)),
        "p": float(p),
    }
    return results


results = getSummary(episodes_table)
results1 = individualComprasion(episodes_table)

print("Mann–Whitney U:", results["mannwhitney_U"])
print("p:", results["p"])
print("Cliff's:", results["cliffs_delta"], "95% CI:", results["cliffs_delta_CI"])
print("Median difference:", results["median_diff_s"], "95% CI:", results["median_diff_CI"])
print("Summary stats (manual):", results["summary_stats"]["manual"])
print("Summary stats (auto):", results["summary_stats"]["auto"])
print("users_paired:", results1["users_paired"])
print("median_diff_s:", results1["median_diff"])
print("median_diff_CI:", results1["median_diff_CI"])
print("p_value:", results1["p"])


