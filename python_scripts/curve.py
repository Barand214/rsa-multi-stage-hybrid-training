import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.cbook


matplotlib.use("Agg")

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

if not hasattr(matplotlib.cbook, "_Stack") and hasattr(matplotlib.cbook, "Stack"):
    matplotlib.cbook._Stack = matplotlib.cbook.Stack


plt.rcParams["font.sans-serif"] = [
    "SimSun",
    "Microsoft YaHei",
    "SimHei",
    "Times New Roman",
    "DejaVu Sans",
]
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["pdf.fonttype"] = 42


COLORS = {
    "red": "#E31A1C",
    "blue": "#1F78B4",
    "green": "#33A02C",
    "purple": "#6A3D9A",
    "orange": "#FF7F00",
    "gray": "#777777",
    "cyan": "#17BECF",
    "pink": "#E377C2",
}


DEFAULT_LOG_DIR = Path("python_scripts/WaveGrad/log/catch_log")
DEFAULT_OUTPUT_ROOT = Path("paper_plots/wavegrad")


def find_latest_wavegrad_log(log_dir=DEFAULT_LOG_DIR):
    log_dir = Path(log_dir)
    candidates = []
    for log_path in log_dir.glob("catch_log_*.json"):
        try:
            log_number = int(log_path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((log_number, log_path))
    if not candidates:
        raise FileNotFoundError(f"No WaveGrad catch logs found in {log_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_wavegrad_json(log_path):
    log_path = Path(log_path)
    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("WaveGrad log must be a JSON object.")
    return data


def _as_float_array(values):
    if values is None:
        return np.asarray([], dtype=np.float64)

    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def _aligned_xy(data, x_key, y_key):
    if x_key not in data or y_key not in data:
        return None, None

    x = _as_float_array(data.get(x_key))
    y = _as_float_array(data.get(y_key))
    n = min(len(x), len(y))
    if n <= 0:
        return None, None

    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None, None
    return x[mask], y[mask]


def rolling_mean(values, window):
    values = _as_float_array(values)
    if values.size == 0:
        return values

    window = max(1, int(window))
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite_values = np.where(np.isfinite(values), values, 0.0)
    finite_count = np.isfinite(values).astype(np.float64)

    csum = np.cumsum(np.insert(finite_values, 0, 0.0))
    ccount = np.cumsum(np.insert(finite_count, 0, 0.0))
    for i in range(values.size):
        start = max(0, i + 1 - window)
        count = ccount[i + 1] - ccount[start]
        if count > 0:
            result[i] = (csum[i + 1] - csum[start]) / count
    return result


def rolling_rate_binary(values, window):
    return rolling_mean(values, window) * 100.0


def _logged_metric_x(data, value_count, interval=100):
    episodes = _as_float_array(data.get("episode_num"))
    matching = episodes[np.isfinite(episodes) & (((episodes + 1) % interval) == 0)]
    if matching.size >= value_count:
        return matching[-value_count:]
    if episodes.size > 0 and np.any(np.isfinite(episodes)):
        first_episode = int(episodes[np.isfinite(episodes)][0])
        first_logged = ((first_episode // interval) + 1) * interval - 1
        return first_logged + np.arange(value_count) * interval
    return np.arange(value_count) * interval + interval - 1


def episode_action_stats(action_sequences, saturation_value=0.85, saturation_tol=1e-6):
    stats = {
        "steps": [],
        "mean": [],
        "mean_abs": [],
        "min": [],
        "max": [],
        "saturation_rate": [],
    }

    for seq in action_sequences or []:
        if not isinstance(seq, list):
            seq = []
        values = _as_float_array(seq)
        values = values[np.isfinite(values)]

        if values.size == 0:
            stats["steps"].append(0.0)
            stats["mean"].append(np.nan)
            stats["mean_abs"].append(np.nan)
            stats["min"].append(np.nan)
            stats["max"].append(np.nan)
            stats["saturation_rate"].append(np.nan)
            continue

        saturated = np.abs(np.abs(values) - float(saturation_value)) <= float(saturation_tol)
        stats["steps"].append(float(values.size))
        stats["mean"].append(float(np.mean(values)))
        stats["mean_abs"].append(float(np.mean(np.abs(values))))
        stats["min"].append(float(np.min(values)))
        stats["max"].append(float(np.max(values)))
        stats["saturation_rate"].append(float(np.mean(saturated) * 100.0))

    return {key: np.asarray(value, dtype=np.float64) for key, value in stats.items()}


def save_figure(fig, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")


def _finish_axis(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", color="grey", alpha=0.3)
    ax.xaxis.grid(False)
    if legend:
        ax.legend(loc="best", fontsize=10, frameon=False)


def _plot_training_return(data, output_dir, window_size):
    x, y = _aligned_xy(data, "episode_num", "return_all")
    if x is None:
        print("Skipped 01_training_return: missing episode_num or return_all")
        return

    y_smooth = rolling_mean(y, window_size)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.plot(x, y, color=COLORS["gray"], linewidth=0.7, alpha=0.28, label="Episode return")
    ax.plot(x, y_smooth, color=COLORS["red"], linewidth=2.0, label=f"Rolling mean ({window_size})")
    _finish_axis(ax, "Training Return", "Episode", "Return")
    fig.tight_layout()
    save_figure(fig, output_dir, "01_training_return")


def _plot_success_rate(data, output_dir, window_size):
    x, goal = _aligned_xy(data, "episode_num", "goal")
    has_goal = x is not None
    rolling_x = None
    rolling_success = None

    if "rolling_success_rate_100" in data:
        rolling_success = _as_float_array(data.get("rolling_success_rate_100"))
        rolling_success = rolling_success[np.isfinite(rolling_success)]
        if rolling_success.size > 0:
            rolling_x = _logged_metric_x(data, rolling_success.size)

    if not has_goal and rolling_success is None:
        print("Skipped 02_success_rate: missing goal and rolling_success_rate_100")
        return

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    if has_goal:
        ax.plot(
            x,
            rolling_rate_binary(goal, window_size),
            color=COLORS["blue"],
            linewidth=2.0,
            label=f"Goal rolling success ({window_size})",
        )
    if rolling_success is not None:
        ax.plot(
            rolling_x,
            rolling_success,
            color=COLORS["red"],
            linewidth=2.0,
            marker="o",
            markersize=3,
            label="Logged rolling success (100)",
        )
    ax.set_ylim(-2, 102)
    _finish_axis(ax, "Training Success Rate", "Episode", "Success rate (%)")
    fig.tight_layout()
    save_figure(fig, output_dir, "02_success_rate")


def _plot_test_success(data, output_dir):
    x, success = _aligned_xy(data, "test_episode", "test_grasp_success_rate")
    score_x, score = _aligned_xy(data, "test_episode", "test_score")
    if x is None and score_x is None:
        print("Skipped 03_test_success: missing test metrics")
        return

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    if x is not None:
        ax.plot(
            x,
            success,
            color=COLORS["red"],
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="Test grasp success",
        )
    if score_x is not None:
        linestyle = "--" if x is not None else "-"
        ax.plot(
            score_x,
            score,
            color=COLORS["blue"],
            linewidth=1.8,
            linestyle=linestyle,
            marker="s",
            markersize=4,
            label="Test score",
        )
    ax.set_ylim(-2, 102)
    _finish_axis(ax, "Evaluation Performance", "Episode", "Score / success rate (%)")
    fig.tight_layout()
    save_figure(fig, output_dir, "03_test_success")


def _plot_losses(data, output_dir, window_size):
    loss_fields = [
        ("loss", "Total loss", COLORS["red"]),
        ("diffusion_loss", "Diffusion loss", COLORS["blue"]),
        ("value_loss", "Value loss", COLORS["green"]),
    ]

    x_base = _as_float_array(data.get("episode_num"))
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=300, sharex=True)
    plotted_any = False

    for field, label, color in loss_fields:
        if field not in data:
            continue
        y = _as_float_array(data.get(field))
        n = min(len(x_base), len(y))
        if n <= 0:
            continue
        x = x_base[:n]
        y = y[:n]
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue
        axes[0].plot(x[mask], rolling_mean(y[mask], window_size), color=color, linewidth=1.7, label=label)
        plotted_any = True

    x, total_loss = _aligned_xy(data, "episode_num", "loss")
    if x is not None:
        axes[1].plot(x, total_loss, color=COLORS["gray"], linewidth=0.7, alpha=0.35, label="Raw total loss")
        axes[1].plot(
            x,
            rolling_mean(total_loss, window_size),
            color=COLORS["red"],
            linewidth=2.0,
            label=f"Rolling total loss ({window_size})",
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("Skipped 04_losses: no loss fields found")
        return

    _finish_axis(axes[0], "Loss Components", "Episode", "Loss", legend=True)
    _finish_axis(axes[1], "Total Loss", "Episode", "Loss", legend=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "04_losses")


def _plot_replay_buffers(data, output_dir):
    x_base = _as_float_array(data.get("episode_num"))
    if x_base.size == 0:
        print("Skipped 05_replay_buffers: missing episode_num")
        return

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    plotted_any = False

    for field, label, color in [
        ("success_replay_size", "Success replay size", COLORS["red"]),
        ("elite_replay_size", "Elite replay size", COLORS["blue"]),
    ]:
        y = _as_float_array(data.get(field))
        n = min(len(x_base), len(y))
        if n > 0:
            ax.plot(x_base[:n], y[:n], color=color, linewidth=1.8, label=label)
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("Skipped 05_replay_buffers: no replay fields found")
        return

    _finish_axis(ax, "Replay Buffers", "Episode", "Transitions", legend=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "05_replay_buffers")


def _plot_safety_penalty(data, output_dir, window_size):
    penalty_x, penalty = _aligned_xy(data, "episode_num", "safety_penalty")
    clip_x, clip_rate = _aligned_xy(data, "episode_num", "safety_clip_rate")
    if penalty_x is None and clip_x is None:
        print("Skipped 06_safety_metrics: missing safety metrics")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=300, sharex=True)
    if penalty_x is not None:
        axes[0].plot(
            penalty_x,
            penalty,
            color=COLORS["gray"],
            linewidth=0.7,
            alpha=0.28,
            label="Episode safety penalty",
        )
        axes[0].plot(
            penalty_x,
            rolling_mean(penalty, window_size),
            color=COLORS["red"],
            linewidth=2.0,
            label=f"Rolling mean ({window_size})",
        )
    if clip_x is not None:
        axes[1].plot(
            clip_x,
            rolling_mean(clip_rate, window_size) * 100.0,
            color=COLORS["blue"],
            linewidth=2.0,
            label="Safety clip rate",
        )
    _finish_axis(axes[0], "Safety Penalty", "Episode", "Penalty", legend=penalty_x is not None)
    _finish_axis(axes[1], "Safety Clip Rate", "Episode", "Clip rate (%)", legend=clip_x is not None)
    fig.tight_layout()
    save_figure(fig, output_dir, "06_safety_metrics")


def _plot_action_statistics(data, output_dir, window_size):
    if "shoulder_actions" not in data and "arm_actions" not in data:
        print("Skipped 07_action_statistics: missing action fields")
        return

    x_base = _as_float_array(data.get("episode_num"))
    if x_base.size == 0:
        max_len = max(len(data.get("shoulder_actions", [])), len(data.get("arm_actions", [])))
        x_base = np.arange(max_len, dtype=np.float64)

    shoulder_stats = episode_action_stats(data.get("shoulder_actions", []))
    arm_stats = episode_action_stats(data.get("arm_actions", []))

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), dpi=300, sharex=True)
    plotted_any = False

    for stats, label, color in [
        (shoulder_stats, "Shoulder", COLORS["red"]),
        (arm_stats, "Arm", COLORS["blue"]),
    ]:
        n = min(len(x_base), len(stats.get("steps", [])))
        if n <= 0:
            continue
        x = x_base[:n]
        axes[0].plot(x, rolling_mean(stats["steps"][:n], window_size), color=color, linewidth=1.8, label=f"{label} steps")
        axes[1].plot(x, rolling_mean(stats["mean"][:n], window_size), color=color, linewidth=1.8, label=f"{label} mean")
        axes[2].plot(x, rolling_mean(stats["mean_abs"][:n], window_size), color=color, linewidth=1.8, label=f"{label} mean abs")
        axes[3].plot(
            x,
            rolling_mean(stats["saturation_rate"][:n], window_size),
            color=color,
            linewidth=1.8,
            label=f"{label} saturation rate",
        )
        axes[1].fill_between(
            x,
            rolling_mean(stats["min"][:n], window_size),
            rolling_mean(stats["max"][:n], window_size),
            color=color,
            alpha=0.12,
            edgecolor="none",
            label=f"{label} min-max",
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("Skipped 07_action_statistics: no valid action statistics")
        return

    _finish_axis(axes[0], "Episode Step Count", "Episode", "Steps", legend=True)
    _finish_axis(axes[1], "Action Mean and Min-Max Range", "Episode", "Action", legend=True)
    _finish_axis(axes[2], "Mean Absolute Action", "Episode", "|Action|", legend=True)
    _finish_axis(axes[3], "Action Saturation Rate", "Episode", "Saturation rate (%)", legend=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "07_action_statistics")


def _plot_logged_rolling_return(data, output_dir):
    rolling_return = _as_float_array(data.get("rolling_mean_return_100"))
    rolling_return = rolling_return[np.isfinite(rolling_return)]
    if rolling_return.size == 0:
        print("Skipped 08_logged_rolling_return: missing rolling_mean_return_100")
        return

    x = _logged_metric_x(data, rolling_return.size)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.plot(
        x,
        rolling_return,
        color=COLORS["red"],
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="Logged rolling mean return (100)",
    )
    _finish_axis(ax, "Logged Rolling Mean Return", "Episode", "Mean return")
    fig.tight_layout()
    save_figure(fig, output_dir, "08_logged_rolling_return")


def _plot_distance_metrics(data, output_dir, window_size):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    plotted_any = False
    for field, label, color in [
        ("final_distance", "Final distance", COLORS["red"]),
        ("min_distance", "Minimum distance", COLORS["blue"]),
    ]:
        x, y = _aligned_xy(data, "episode_num", field)
        if x is None:
            continue
        ax.plot(x, rolling_mean(y, window_size), color=color, linewidth=1.8, label=label)
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        print("Skipped 09_distance_metrics: missing distance fields")
        return
    _finish_axis(ax, "Catch Distance", "Episode", "Distance", legend=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "09_distance_metrics")


def _plot_policy_lr(data, output_dir):
    x, y = _aligned_xy(data, "episode_num", "policy_lr")
    if x is None:
        print("Skipped 10_policy_lr: missing policy_lr")
        return
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.plot(x, y, color=COLORS["purple"], linewidth=1.8, label="Policy learning rate")
    _finish_axis(ax, "Policy Learning Rate", "Episode", "Learning rate", legend=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "10_policy_lr")


def _plot_candidate_scoring(data, output_dir, window_size):
    fields = [
        ("candidate_score_mean", "Candidate score mean", COLORS["red"]),
        ("selected_candidate_index_mean", "Selected candidate index mean", COLORS["blue"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    plotted_any = False
    for field, label, color in fields:
        x, y = _aligned_xy(data, "episode_num", field)
        if x is None:
            continue
        ax.plot(x, rolling_mean(y, window_size), color=color, linewidth=1.8, label=label)
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        print("Skipped 11_candidate_scoring: candidate scoring is not enabled")
        return
    _finish_axis(ax, "Candidate Scoring", "Episode", "Score / index", legend=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "11_candidate_scoring")


def plot_all_wavegrad_metrics(log_path, output_dir, window_size=100):
    data = load_wavegrad_json(log_path)
    output_dir = Path(output_dir)

    print(f"Loaded WaveGrad log: {Path(log_path)}")
    print(f"Output directory: {output_dir}")
    print(f"Available fields: {', '.join(data.keys())}")

    _plot_training_return(data, output_dir, window_size)
    _plot_success_rate(data, output_dir, window_size)
    _plot_test_success(data, output_dir)
    _plot_losses(data, output_dir, window_size)
    _plot_replay_buffers(data, output_dir)
    _plot_safety_penalty(data, output_dir, window_size)
    _plot_action_statistics(data, output_dir, window_size)
    _plot_logged_rolling_return(data, output_dir)
    _plot_distance_metrics(data, output_dir, window_size)
    _plot_policy_lr(data, output_dir)
    _plot_candidate_scoring(data, output_dir, window_size)


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot WaveGrad catch training logs.")
    parser.add_argument(
        "--log",
        default=None,
        help="Path to a WaveGrad JSON log. The latest catch log is used when omitted.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to paper_plots/wavegrad/<experiment name>.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Experiment name used for the default output directory.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling window size for smoothed curves.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    selected_log = Path(args.log) if args.log else find_latest_wavegrad_log()
    experiment_name = args.name or selected_log.stem
    selected_output = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / experiment_name
    plot_all_wavegrad_metrics(selected_log, selected_output, args.window)
