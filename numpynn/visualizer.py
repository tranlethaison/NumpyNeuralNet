import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.make_plots()

    def make_plots(self):
        self.fig, self.axes = plt.subplots(2, 1, sharex=True)

        # Losses
        self.axes[0].set_ylabel("Loss")
        self.loss_id = 0

        self.loss_data = [None] * self.n_epochs
        self.loss, = self.axes[0].plot(self.loss_data, label="loss")

        self.val_loss_data = [None] * self.n_epochs
        self.val_loss, = self.axes[0].plot(self.val_loss_data, label="val_loss")

        # Accuracy
        self.axes[1].set_ylabel("Accuracy")
        self.accu_id = 0

        self.accu_data = [None] * self.n_epochs
        self.accu, = self.axes[1].plot(self.accu_data, label="accuracy")

        self.val_accu_data = [None] * self.n_epochs
        self.val_accu, = self.axes[1].plot(self.val_accu_data, label="val_accuracy")

        # Common
        self.axes[-1].set_xlim(0, self.n_epochs - 1)
        self.axes[-1].set_xlabel("Epoch")

        for ax in self.axes:
            ax.grid(True)
            ax.legend()

    def update_loss(self, loss, val_loss):
        self.loss_data[self.loss_id] = loss
        self.loss.set_ydata(self.loss_data)

        self.val_loss_data[self.loss_id] = val_loss
        self.val_loss.set_ydata(self.val_loss_data)

        self.loss_id += 1
        self.axes[0].relim()
        self.axes[0].autoscale_view()

    def update_accu(self, accu, val_accu):
        self.accu_data[self.accu_id] = accu
        self.accu.set_ydata(self.accu_data)

        self.val_accu_data[self.accu_id] = val_accu
        self.val_accu.set_ydata(self.val_accu_data)

        self.accu_id += 1
        self.axes[1].relim()
        self.axes[1].autoscale_view()
