
def scale_axis_to_zero(x_axis, y_axis, ax):
    ymin = y_axis.min()
    ymax = y_axis.max()

    y_bottom = ymin if ymin < 0 else 0
    y_top = ymax if ymax > 0 else 0

    padding = (y_top - y_bottom) * 0.1
    ax.set_ylim(bottom=y_bottom - padding, top=y_top + padding)

    # X-Achse auf ganze Zahlen beschränken
    max_epoch = int(x_axis.max())  # Maximalwert der EPOCH_COL
    ax.set_xticks(range(0, max_epoch + 1))

def set_labels(self, ax):
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid()