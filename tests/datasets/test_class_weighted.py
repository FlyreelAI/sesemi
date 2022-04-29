import pytest

from sesemi.datasets import dataset


@pytest.mark.parametrize(
    "num_classes,weights",
    [
        (3, [1] * 3),
        (5, [1] * 5),
        (3, [0.1, 0.3, 0.6]),
        (3, None),
    ],
)
def test_class_weighted(num_classes, weights):
    dst = [(0, i) for i in range(num_classes)]

    class_weighted_dataset = dataset(
        name="class_weighted",
        root="",
        dataset=dst,
        weights=weights,
        seed=42,
    )

    labels = [x[1] for _, x in zip(range(1000), class_weighted_dataset)]

    counts = [0] * num_classes
    for l in labels:
        counts[l] += 1

    count_distribution = [x / len(labels) for x in counts]

    weights = weights or ([1] * num_classes)
    weight_sum = sum(weights)
    weight_distribution = [x / weight_sum for x in weights]

    tolerance = 0.1
    max_error = max(
        [abs(x - y) for x, y in zip(count_distribution, weight_distribution)]
    )
    assert max_error <= tolerance
