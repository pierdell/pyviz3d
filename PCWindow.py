import time
from threading import Thread
from typing import Optional

import numpy as np

from opengl.pygame_engine import ExplorationEngine, PointCloudModel


class PCWindow:
    """
    Opens a PyGame Window to display a PointCloud
    """

    def __init__(self,
                 pointcloud_size: int = 3,
                 default_pc_color: np.ndarray = np.array([[0.0, 0.6, 1.0]])):
        self.engine = ExplorationEngine(point_size=pointcloud_size)
        self.engine.point_size = pointcloud_size
        self.default_pc_color = default_pc_color

        self.pc_model_id = 1
        self.pc_model = PointCloudModel(np.zeros((1, 3), dtype=np.float32), default_color=default_pc_color)
        self.engine.add_model(self.pc_model_id, self.pc_model)
        self.window_thread = Thread(target=lambda: self.engine.main_loop())

    def init(self):
        """Initialize the window"""
        self.window_thread.start()

    def close(self):
        """Closes the Pygame window"""
        self.window_thread.join()

    def update_model(self, pointcloud: np.ndarray, color: Optional[np.ndarray] = None):
        """Updates the PointCloud model held by the window"""
        self.pc_model.set_pointcloud(pointcloud, color)
        self.engine.update_model(self.pc_model_id)

    # TODO
    # def update_camera(self, position: np.ndarray):
    #     """Updates the camera position programmatically"""
    #
    #     pass

    def is_alive(self):
        """Whether the Current window is still alive"""
        return self.window_thread.is_alive()



