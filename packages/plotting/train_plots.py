from matplotlib import pyplot as plt


def plot_history_tight(history):
    train_history_dict = history.train_history.copy()
    del train_history_dict['epoch']
    val_history_dict = history.val_history.copy()
    del val_history_dict['epoch']
    plt.figure(figsize=(8, 6))
    for metric_name, values in train_history_dict.items():
        plt.plot(values, label=f"Train {metric_name}", linewidth=2)
    for metric_name, values in val_history_dict.items():
        plt.plot(values, label=f"Val {metric_name}", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_history_extended(history):
    train_history_dict = history.train_history.copy()
    del train_history_dict['epoch']
    val_history_dict = history.val_history.copy()
    del val_history_dict['epoch']
    num_metrics = len(train_history_dict)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))

    if num_metrics == 1:
        axes = [axes]

    for i, (metric_name, train_values) in enumerate(train_history_dict.items()):
        axes[i].plot(train_values, label=f"Train {metric_name}", linewidth=2)
        if metric_name in val_history_dict:
            val_values = val_history_dict[metric_name]
            axes[i].plot(val_values, label=f"Val {metric_name}", linewidth=2)

        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f"{metric_name} Over Time")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()
