"""End-to-end test of the live prediction path (Predictor) without the network.

Feeds synthetic candles through the exact same Predictor class that ai_bot.py
uses, so we validate the live code path — not just the training script.
"""
import csv

from ai_bot import Predictor


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            if r and r[0].lstrip("-").isdigit():
                rows.append([int(float(r[0])), float(r[1]), float(r[2]),
                             float(r[3]), float(r[4])])
    return rows


def run(model_path, csv_path, label, split=0.6):
    """Evaluate OUT-OF-SAMPLE: fit on the first `split` of the data, score the rest."""
    pred = Predictor(model_path)
    candles = load(csv_path)
    cut = int(len(candles) * split)

    # retrain on the first part only, so the evaluation is honest
    from features import compute_features, features_to_vector
    from sklearn.ensemble import GradientBoostingClassifier
    import numpy as np
    X, y = [], []
    for i in range(cut - 1):
        f = compute_features(candles, i)
        if f is None:
            continue
        X.append(features_to_vector(f))
        y.append(1 if candles[i + 1][2] > candles[i][2] else 0)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, subsample=0.9,
                                       random_state=42).fit(np.array(X), np.array(y))
    pred.model = model

    correct = graded = signals = 0
    for i in range(cut, len(candles) - 1):
        res = pred.predict(candles[:i + 1])
        if res is None:
            continue
        action, conf = res
        if conf < 0.55:
            continue
        signals += 1
        won = ((action == "UP" and candles[i + 1][2] > candles[i][2]) or
               (action == "DOWN" and candles[i + 1][2] < candles[i][2]))
        graded += 1
        correct += int(won)
    acc = correct / graded * 100 if graded else 0.0
    print(f"{label:12} signals={signals:4}  graded={graded:4}  hit-rate={acc:5.1f}%")
    return acc


print("live-path (Predictor) evaluation on synthetic data:\n")
run("out/random.pkl", "data/synthetic_random.csv", "random")
run("out/momentum.pkl", "data/synthetic_momentum.csv", "momentum")
print("\nexpected: random ~50% (coin flip), momentum clearly higher.")
