import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd 

from types import SimpleNamespace
import pandas as pd 

# data = pd.read_csv('Rotem_model/data/full_muvment_clean/08_Feb_2024_10_27_full_movment_clean.csv')

# import yaml

# with open('Rotem_model/config.yaml', 'r') as f:
#     config = yaml.safe_load(f)
# config =SimpleNamespace(**config)
# locations = data[[
#     'MCx','MCy', 'MCz','MSx', 'MSy', 'MSz',
#     'MEx', 'MEy', 'MEz',
#     'MWx', 'MWy', 'MWz'
#     ]]
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

def draw_arm(df_pred, df_true):
    # Initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Adjust these limits according to your data range
    all_data = pd.concat([df_pred, df_true])
    ax.set_xlim([all_data[['MSx', 'MEx', 'MWx']].min().min(), all_data[['MSx', 'MEx', 'MWx']].max().max()])
    ax.set_ylim([all_data[['MSy', 'MEy', 'MWy']].min().min(), all_data[['MSy', 'MEy', 'MWy']].max().max()])
    ax.set_zlim([all_data[['MSz', 'MEz', 'MWz']].min().min(), all_data[['MSz', 'MEz', 'MWz']].max().max()])

    # Lines to connect points for predictions and true data
    line_pred, = ax.plot([], [], [], 'ro-', label='Predicted')
    line_true, = ax.plot([], [], [], 'bo-', label='True')

    def update(frame):
        # Extract points for the current frame for predictions
        frame = frame + 1000
        shoulder_pred = (df_pred.at[frame, 'MSx'], df_pred.at[frame, 'MSy'], df_pred.at[frame, 'MSz'])
        elbow_pred = (df_pred.at[frame, 'MEx'], df_pred.at[frame, 'MEy'], df_pred.at[frame, 'MEz'])
        wrist_pred = (df_pred.at[frame, 'MWx'], df_pred.at[frame, 'MWy'], df_pred.at[frame, 'MWz'])

        # Extract points for the current frame for true data
        shoulder_true = (df_true.at[frame, 'MSx'], df_true.at[frame, 'MSy'], df_true.at[frame, 'MSz'])
        elbow_true = (df_true.at[frame, 'MEx'], df_true.at[frame, 'MEy'], df_true.at[frame, 'MEz'])
        wrist_true = (df_true.at[frame, 'MWx'], df_true.at[frame, 'MWy'], df_true.at[frame, 'MWz'])

        # Update line data for predictions and true data
        line_pred.set_data([shoulder_pred[0], elbow_pred[0], wrist_pred[0]], [shoulder_pred[1], elbow_pred[1], wrist_pred[1]])
        line_pred.set_3d_properties([shoulder_pred[2], elbow_pred[2], wrist_pred[2]])

        line_true.set_data([shoulder_true[0], elbow_true[0], wrist_true[0]], [shoulder_true[1], elbow_true[1], wrist_true[1]])
        line_true.set_3d_properties([shoulder_true[2], elbow_true[2], wrist_true[2]])

        ax.view_init(elev=30, azim=45)
        print(frame)
        return line_pred, line_true

    ax.legend()
    ani = FuncAnimation(fig, update, frames=2000, blit=True, repeat=False, interval=10)
    # Save the animation
    # Save the animation
    plt.show()
    writer = FFMpegWriter(fps=100, metadata=dict(artist='video'), bitrate=1800)
    ani.save('predsVStrue.mp4', writer=writer)
    
    # ani.save('predsVStrue.mp4',writer=writer)


    plt.close()




df_pred = pd.read_csv('Rotem_model/df_preds.csv')
df_true = pd.read_csv('Rotem_model/df_true.csv')

draw_arm(df_pred, df_true)