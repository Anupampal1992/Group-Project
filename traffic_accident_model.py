"""Baseline accident severity classifier using pure Python (no external deps)."""
from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DATA_PATH = Path("Traffic Accident Severity Predictor Dataset.csv")
RANDOM_SEED = 42
TEST_RATIO = 0.2
TARGET = "Accident_Severity"

CATEGORICAL_FEATURES = [
    "Weather",
    "Road_Type",
    "Time_of_Day",
    "Road_Condition",
    "Vehicle_Type",
    "Road_Light_Condition",
]

NUMERIC_FEATURES = [
    "Traffic_Density",
    "Speed_Limit",
    "Number_of_Vehicles",
    "Driver_Alcohol",
    "Driver_Age",
    "Driver_Experience",
]

IGNORED_FEATURES = {"Accident"}


@dataclass
class Dataset:
    features: List[Dict[str, str]]
    targets: List[str]


@dataclass
class GaussianStats:
    mean: float
    variance: float


@dataclass
class Model:
    class_priors: Dict[str, float]
    gaussian_stats: Dict[str, Dict[str, GaussianStats]]
    categorical_counts: Dict[str, Dict[str, Counter]]
    categorical_totals: Dict[str, Dict[str, int]]
    category_vocab: Dict[str, List[str]]


def load_dataset(path: Path) -> Dataset:
    features: List[Dict[str, str]] = []
    targets: List[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if any(row.get(col, "") == "" for col in [TARGET, *CATEGORICAL_FEATURES, *NUMERIC_FEATURES]):
                continue
            features.append({k: v for k, v in row.items() if k not in IGNORED_FEATURES and k != TARGET})
            targets.append(row[TARGET])
    return Dataset(features, targets)


def stratified_split(dataset: Dataset, test_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    random.seed(seed)
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        buckets[label].append(idx)

    train_idx: List[int] = []
    test_idx: List[int] = []
    for label, indices in buckets.items():
        random.shuffle(indices)
        split = int(len(indices) * (1 - test_ratio))
        train_idx.extend(indices[:split])
        test_idx.extend(indices[split:])

    def subset(indices: Iterable[int]) -> Dataset:
        return Dataset(
            features=[dataset.features[i] for i in indices],
            targets=[dataset.targets[i] for i in indices],
        )

    return subset(train_idx), subset(test_idx)


def compute_gaussian_stats(values: List[float]) -> GaussianStats:
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    variance = max(variance, 1e-6)
    return GaussianStats(mean, variance)


def train_naive_bayes(train: Dataset) -> Model:
    class_counts = Counter(train.targets)
    total = len(train.targets)
    class_priors = {label: count / total for label, count in class_counts.items()}

    gaussian_stats: Dict[str, Dict[str, GaussianStats]] = defaultdict(dict)
    categorical_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    categorical_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    category_vocab: Dict[str, set] = {feature: set() for feature in CATEGORICAL_FEATURES}

    for row, label in zip(train.features, train.targets):
        for feature in CATEGORICAL_FEATURES:
            value = row[feature]
            category_vocab[feature].add(value)
            categorical_counts[label][feature][value] += 1
            categorical_totals[label][feature] += 1

    for label in class_counts:
        for feature in NUMERIC_FEATURES:
            values = [float(row[feature]) for row, y in zip(train.features, train.targets) if y == label]
            gaussian_stats[label][feature] = compute_gaussian_stats(values)

    category_vocab_list = {feature: sorted(values) for feature, values in category_vocab.items()}

    return Model(
        class_priors=class_priors,
        gaussian_stats=gaussian_stats,
        categorical_counts=categorical_counts,
        categorical_totals=categorical_totals,
        category_vocab=category_vocab_list,
    )


def gaussian_log_prob(x: float, stats: GaussianStats) -> float:
    return -0.5 * math.log(2 * math.pi * stats.variance) - ((x - stats.mean) ** 2) / (2 * stats.variance)


def predict_row(model: Model, row: Dict[str, str]) -> str:
    best_label = None
    best_log_prob = -float("inf")

    for label, prior in model.class_priors.items():
        log_prob = math.log(prior)
        for feature in NUMERIC_FEATURES:
            log_prob += gaussian_log_prob(float(row[feature]), model.gaussian_stats[label][feature])
        for feature in CATEGORICAL_FEATURES:
            value = row[feature]
            count = model.categorical_counts[label][feature][value]
            total = model.categorical_totals[label][feature]
            vocab_size = len(model.category_vocab[feature])
            log_prob += math.log((count + 1) / (total + vocab_size))
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_label = label
    return best_label or ""


def classification_report(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
    labels = sorted(set(y_true))
    report: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        report[label] = {"precision": precision, "recall": recall, "f1": f1}
    return report


def confusion_matrix(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
    labels = sorted(set(y_true))
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for yt, yp in zip(y_true, y_pred):
        matrix[yt][yp] += 1
    return matrix


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    train, test = stratified_split(dataset, TEST_RATIO, RANDOM_SEED)
    model = train_naive_bayes(train)

    predictions = [predict_row(model, row) for row in test.features]
    accuracy = sum(1 for yt, yp in zip(test.targets, predictions) if yt == yp) / len(test.targets)
    report = classification_report(test.targets, predictions)
    matrix = confusion_matrix(test.targets, predictions)

    macro_f1 = sum(metrics["f1"] for metrics in report.values()) / len(report)

    print("Rows used:", len(dataset.targets))
    print("Train size:", len(train.targets))
    print("Test size:", len(test.targets))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Per-class metrics:")
    for label, metrics in report.items():
        print(f"  {label}: precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}")
    print("Confusion matrix:")
    for label, row in matrix.items():
        print(f"  {label}: {row}")


if __name__ == "__main__":
    main()
