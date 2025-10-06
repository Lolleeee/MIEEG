from matplotlib import pyplot as plt


def plot_history_tight(history):
    plt.figure(figsize=(8, 6))
    for metric_name, values in history.train_history.items():
        plt.plot(values, label=f"Train {metric_name}", linewidth=2)
    for metric_name, values in history.val_history.items():
        plt.plot(values, label=f"Val {metric_name}", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_history_extended(history):
    num_metrics = len(history.train_history)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))

    if num_metrics == 1:
        axes = [axes]

    for i, (metric_name, train_values) in enumerate(history.train_history.items()):
        axes[i].plot(train_values, label=f"Train {metric_name}", linewidth=2)
        if metric_name in history.val_history:
            val_values = history.val_history[metric_name]
            axes[i].plot(val_values, label=f"Val {metric_name}", linewidth=2)

        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f"{metric_name} Over Time")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()
