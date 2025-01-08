from time import perf_counter
from obliquetree import Classifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

# Test parameters
n_columns = 25
n_samples = [10000, 50000, 100000, 250000]
n_uniques = [5, 10, 15, 25]


def create_comparison_plot_int(n_samples, n_uniques, n_columns=25, figsize=(10, 6)):
    # Initialize the plot
    plt.figure(figsize=figsize)

    # Calculate bar positions
    bar_width = 0.1
    r = np.arange(len(n_samples))

    # Colors for different n_unique values
    colors = plt.cm.Set3(np.linspace(0, 1, len(n_uniques)))

    # Create bars for each n_unique value
    for idx, n_unique in enumerate(n_uniques):
        times_oblique = []
        times_sklearn = []

        # Collect performance data
        for n_sample in n_samples:
            # Generate random data
            X = np.random.randint(0, n_unique, (n_sample, n_columns))
            y = np.random.randint(0, 2, n_sample)

            # Time obliquetree Tree
            start = perf_counter()
            tree1 = Classifier(use_oblique=False)
            tree1.fit(X, y)
            times_oblique.append(perf_counter() - start)

            # Time scikit-learn
            start = perf_counter()
            tree2 = DecisionTreeClassifier()
            tree2.fit(X, y)
            times_sklearn.append(perf_counter() - start)

        # Plot bars
        pos_oblique = r + idx * bar_width * 2
        pos_sklearn = r + idx * bar_width * 2 + bar_width

        plt.bar(
            pos_oblique,
            times_oblique,
            bar_width,
            label=f"obliquetree (Unique values per column={n_unique})",
            color=colors[idx],
            alpha=0.8,
        )
        plt.bar(
            pos_sklearn,
            times_sklearn,
            bar_width,
            label=f"scikit-learn (Unique values per column={n_unique})",
            color=colors[idx],
            alpha=0.8,
            hatch="//",
        )

        # Add value labels
        for i in range(len(times_oblique)):
            plt.text(
                pos_oblique[i],
                times_oblique[i],
                f"{times_oblique[i]:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )
            plt.text(
                pos_sklearn[i],
                times_sklearn[i],
                f"{times_sklearn[i]:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )

    # Customize plot
    plt.xlabel("Number of Samples")
    plt.ylabel("Fit Duration (seconds)")
    plt.title(f"Performance Comparison Integer Columns")
    plt.xticks(r + bar_width * (len(n_uniques) - 0.5), [str(x) for x in n_samples])
    # Changed legend position to upper left inside the plot
    plt.legend(loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.ylim(0, max(max(times_oblique), max(times_sklearn)) * 1.10)
    plt.tight_layout()

    return plt.gcf()


def create_comparison_plot_float(n_samples, n_columns=25, figsize=(10, 6)):
    # Float comparison plot code remains the same
    # Initialize the plot
    plt.figure(figsize=figsize)

    # Calculate bar positions
    bar_width = 0.35
    r = np.arange(len(n_samples))

    times_oblique = []
    times_sklearn = []

    # Collect performance data
    for n_sample in n_samples:
        # Generate random data from normal distribution
        X = np.random.normal(0, 1, (n_sample, n_columns))
        y = np.random.randint(0, 2, n_sample)

        # Time obliquetree Tree
        start = perf_counter()
        tree1 = Classifier(use_oblique=False)
        tree1.fit(X, y)
        times_oblique.append(perf_counter() - start)

        # Time scikit-learn
        start = perf_counter()
        tree2 = DecisionTreeClassifier()
        tree2.fit(X, y)
        times_sklearn.append(perf_counter() - start)

    # Plot bars
    plt.bar(
        r - bar_width / 2,
        times_oblique,
        bar_width,
        label="obliquetree",
        color="skyblue",
        alpha=0.8,
    )
    plt.bar(
        r + bar_width / 2,
        times_sklearn,
        bar_width,
        label="scikit-learn",
        color="lightcoral",
        alpha=0.8,
        hatch="//",
    )

    # Add value labels
    for i in range(len(times_oblique)):
        plt.text(
            r[i] - bar_width / 2,
            times_oblique[i],
            f"{times_oblique[i]:.3f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )
        plt.text(
            r[i] + bar_width / 2,
            times_sklearn[i],
            f"{times_sklearn[i]:.3f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )

    # Customize plot
    plt.xlabel("Number of Samples")
    plt.ylabel("Fit Duration (seconds)")
    plt.title(f"Performance Comparison Float Columns")
    plt.xticks(r, [str(x) for x in n_samples])
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)

    plt.ylim(0, max(max(times_oblique), max(times_sklearn)) * 1.10)
    plt.tight_layout()

    return plt.gcf()


# Create and save plot
fig = create_comparison_plot_float(n_samples, n_columns)
plt.savefig("performance_comparison_float.png", dpi=300, bbox_inches="tight")
plt.close()

fig = create_comparison_plot_int(n_samples, n_uniques, n_columns)
plt.savefig("performance_comparison_int.png", dpi=300, bbox_inches="tight")
plt.close()