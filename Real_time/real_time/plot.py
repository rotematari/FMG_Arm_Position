import matplotlib.pyplot as plt
import numpy as np

class DynamicPlot:
    def __init__(self):
        """Initialize the dynamic 3D plot."""
        self.fig = plt.figure(figsize=[10, 10])
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.line_pred, = self.ax.plot([], [], [], 'r-', label='Prediction')
        self.line_true, = self.ax.plot([], [], [], 'b-', label='True')
        plt.legend()

        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        self.ax.set_xlim([-40, 60])
        self.ax.set_ylim([-50, 50])
        self.ax.set_zlim([-50, 60])
        
        # Set the view angle
        self.ax.view_init(elev=30, azim=10)

        plt.show(block=False)
        plt.pause(0.00001)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.line_pred)
        self.ax.draw_artist(self.line_true)
        self.fig.canvas.blit(self.fig.bbox)

    def update_plot(self, new_data_pred: np.ndarray, new_data_true: dict) -> None:
        """Update the 3D plot with new data."""
        new_data_pred = new_data_pred.reshape((-1))
        elbow_pred, wrist_pred = new_data_pred[:3]*100, new_data_pred[3:]*100
        shoulder_pred =[-14.62,-32.0,57.11]
        
        shoulder_true = tuple(x * 100 for x in new_data_true['shoulder'][0])
        elbow_true = tuple(x * 100 for x in new_data_true['elbow'][0])
        wrist_true = tuple(x * 100 for x in new_data_true['wrist'][0])

        self.line_pred.set_data([shoulder_pred[0], elbow_pred[0], wrist_pred[0]], [shoulder_pred[1], elbow_pred[1], wrist_pred[1]])
        self.line_pred.set_3d_properties([shoulder_pred[2], elbow_pred[2], wrist_pred[2]])

        self.line_true.set_data([shoulder_true[0], elbow_true[0], wrist_true[0]], [shoulder_true[1], elbow_true[1], wrist_true[1]])
        self.line_true.set_3d_properties([shoulder_true[2], elbow_true[2], wrist_true[2]])

        self.fig.canvas.restore_region(self.bg)
        self.ax.draw_artist(self.line_pred)
        self.ax.draw_artist(self.line_true)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()