import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


SMALL_BATCH_EXPERIMENTS = {
    "cartpole": "Vanilla PG",
    "cartpole_rtg": "Reward-to-go",
    "cartpole_na": "Adv. norm",
    "cartpole_rtg_na": "RTG + adv. norm",
}

LARGE_BATCH_EXPERIMENTS = {
    "cartpole_lb": "Vanilla PG",
    "cartpole_lb_rtg": "Reward-to-go",
    "cartpole_lb_na": "Adv. norm",
    "cartpole_lb_rtg_na": "RTG + adv. norm",
}

COLORS = {
    "cartpole": "#4C78A8",
    "cartpole_rtg": "#F58518",
    "cartpole_na": "#54A24B",
    "cartpole_rtg_na": "#B279A2",
    "cartpole_lb": "#4C78A8",
    "cartpole_lb_rtg": "#F58518",
    "cartpole_lb_na": "#54A24B",
    "cartpole_lb_rtg_na": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CartPole learning curves from CS285 HW2 log.csv files."
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("exp"),
        help="Directory containing experiment run folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plot images will be saved.",
    )
    parser.add_argument(
        "--x-key",
        default="Train_EnvstepsSoFar",
        help="CSV column to use for the x-axis.",
    )
    parser.add_argument(
        "--y-key",
        default="Train_AverageReturn",
        help="CSV column to use for the y-axis.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image resolution.",
    )
    return parser.parse_args()


def experiment_name(run_dir: Path) -> str | None:
    prefix = "CartPole-v0_"
    marker = "_sd"
    name = run_dir.name
    if not name.startswith(prefix) or marker not in name:
        return None
    return name[len(prefix) : name.index(marker)]


def find_latest_logs(exp_dir: Path, expected_names: Iterable[str]) -> dict[str, Path]:
    expected = set(expected_names)
    matches: dict[str, Path] = {}

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        exp_name = experiment_name(run_dir)
        if exp_name not in expected:
            continue
        log_path = run_dir / "log.csv"
        if not log_path.is_file():
            continue
        if exp_name not in matches or run_dir.name > matches[exp_name].parent.name:
            matches[exp_name] = log_path

    missing = sorted(expected - set(matches))
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing CartPole log.csv files for: {missing_list}")

    return matches


def read_curve(log_path: Path, x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []

    with log_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{log_path} has no CSV header")
        for key in (x_key, y_key):
            if key not in reader.fieldnames:
                raise KeyError(f"{log_path} is missing required column: {key}")

        for row in reader:
            if row[x_key] == "" or row[y_key] == "":
                continue
            xs.append(float(row[x_key]))
            ys.append(float(row[y_key]))

    return xs, ys


def style_axes(ax: plt.Axes, title: str, x_label: str, y_label: str) -> None:
    ax.set_title(title, fontsize=16, weight="bold", pad=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(0, 210)
    ax.axhline(200, color="#222222", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.grid(True, which="major", color="#D8DEE9", linewidth=0.9)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#9AA4B2")

    ax.tick_params(colors="#354052", labelsize=10)
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD2D9",
        framealpha=0.92,
        fontsize=10,
    )


def plot_group(
    logs: dict[str, Path],
    labels: dict[str, str],
    title: str,
    output_path: Path,
    x_key: str,
    y_key: str,
    dpi: int,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFE")

    for exp_name, label in labels.items():
        xs, ys = read_curve(logs[exp_name], x_key, y_key)
        ax.plot(
            xs,
            ys,
            label=label,
            color=COLORS[exp_name],
            linewidth=2.6,
            alpha=0.95,
        )

    style_axes(
        ax,
        title=title,
        x_label="Environment steps",
        y_label=y_key.replace("_", " "),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    small_logs = find_latest_logs(args.exp_dir, SMALL_BATCH_EXPERIMENTS)
    large_logs = find_latest_logs(args.exp_dir, LARGE_BATCH_EXPERIMENTS)

    small_path = args.out_dir / "cartpole_small_batch.png"
    large_path = args.out_dir / "cartpole_large_batch.png"

    plot_group(
        small_logs,
        SMALL_BATCH_EXPERIMENTS,
        "CartPole Small Batch (b=1000)",
        small_path,
        args.x_key,
        args.y_key,
        args.dpi,
    )
    plot_group(
        large_logs,
        LARGE_BATCH_EXPERIMENTS,
        "CartPole Large Batch (b=4000)",
        large_path,
        args.x_key,
        args.y_key,
        args.dpi,
    )

    print(f"Saved {small_path}")
    print(f"Saved {large_path}")


if __name__ == "__main__":
    main()
