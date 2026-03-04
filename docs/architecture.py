from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_box(ax, xy, text, box_color="#1f77b4"):
    x, y = xy
    width, height = 3.4, 0.9
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.2,rounding_size=0.15",
        linewidth=1.5,
        edgecolor="black",
        facecolor=box_color,
        alpha=0.9,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        wrap=True,
    )
    return x + width / 2, y  # return center-top for arrow start


def draw_arrow(ax, start_xy, end_xy):
    ax.annotate(
        "",
        xy=end_xy,
        xytext=start_xy,
        arrowprops=dict(arrowstyle="->", lw=1.6, color="black"),
    )


def plot_architecture(save_path: Path | None = None) -> None:
    """
    Generate a system architecture diagram for the LSTM autoencoder-based
    network anomaly detection system / NWDAF analytics pipeline.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10)
    ax.axis("off")

    y_positions = [9, 8, 7, 6, 5, 4, 3, 2]
    x_center = 1.3

    elements = [
        "Network Traffic Dataset",
        "Feature Extraction\n(Preprocessing, Normalization)",
        "Sequence Generator\n(Time-windowed LSTM sequences)",
        "LSTM Autoencoder Model\n(Encoder–Decoder)",
        "Reconstruction Error\nCalculation",
        "Threshold Detection\n(Percentile / Statistical)",
        "Anomaly Alerts\n(Normal vs Attack)",
        "NWDAF Analytics Dashboard\n(Visualization & Reporting)",
    ]

    centers = []
    for y, label in zip(y_positions, elements):
        cx, top_y = draw_box(ax, (x_center, y - 0.45), label)
        centers.append((cx, top_y))

    # Draw arrows between consecutive components
    for i in range(len(centers) - 1):
        start = (centers[i][0], centers[i][1])
        end = (centers[i + 1][0], centers[i + 1][1] + 0.9)
        draw_arrow(ax, start, end)

    ax.set_title(
        "LSTM Autoencoder-based Network Anomaly Detection\nZero Touch NWDAF System Architecture",
        fontsize=12,
        pad=20,
    )

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Architecture diagram saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Default: save PNG into docs directory for inclusion in reports.
    output = Path(__file__).with_name("architecture_diagram.png")
    plot_architecture(save_path=output)

