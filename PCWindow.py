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

        self.pc_models = dict()
        self.window_thread = None

    def init(self):
        """Initialize the window"""
        if self.window_thread is None:
            self.window_thread = Thread(target=lambda: self.engine.main_loop())
        self.window_thread.start()

    def close(self, force_close: bool = False):
        """Closes the Pygame window"""
        if force_close:
            self.engine.force_close()
        self.window_thread.join()
        self.window_thread = None

    def update_model(self, pc_id: int, pointcloud: np.ndarray,
                     color: Optional[np.ndarray] = None,
                     default_color: Optional[np.ndarray] = None):
        """Updates the PointCloud model held by the window"""
        if pc_id not in self.pc_models:
            if default_color is None:
                default_color = self.default_pc_color
            pc_model = PointCloudModel(pointcloud, color, default_color=default_color)
            self.pc_models[pc_id] = pc_model
            self.engine.add_model(pc_id, pc_model)
            return

        self.pc_models[pc_id].set_pointcloud(pointcloud, color)
        self.engine.update_model(pc_id)

    def update_camera(self, camera_pose: np.ndarray):
        """Updates the camera position programmatically"""
        self.engine.update_camera(camera_pose)

    def is_alive(self):
        """Whether the Current window is still alive"""
        return self.window_thread.is_alive()


if __name__ == "__main__":

    window = PCWindow()
    try:
        window.init()
        for _ in range(1000):
            time.sleep(1)
            window.update_model(0, np.random.randn(1000, 3))
            if not window.is_alive():
                print("Dead window")
                break
        window.close()
    except (KeyboardInterrupt, Exception):
        window.close(True)
