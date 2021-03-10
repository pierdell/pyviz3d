import unittest
import multiprocessing as mp

from viz3d.opengl.pygame_engine import PointCloudModelData, farray
from viz3d.engineprocess import *
from viz3d.window import OpenGLWindow, CamerasModelData


class EngineTestCase(unittest.TestCase):
    def test_engine(self):
        engine = ExplorationEngine()
        engine.add_model(0, PointCloudModelData(xyz=np.random.randn(10000, 3), default_color=None))
        engine.main_loop()

        self.assertEqual(True, True)

    def test_instanciation(self):
        engine = ExplorationEngine()
        instances = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(3, axis=0)
        instances[0, 0, 3] = -10
        instances[1, 0, 3] = 0
        instances[2, 0, 3] = 10

        engine.add_model(0, PointCloudModelData(xyz=np.random.randn(10000, 3),
                                                default_color=None,
                                                instance_model_to_world=instances))
        engine.main_loop()
        self.assertEqual(True, True)

    def test_process_communication(self):
        ctxt = mp.get_context('spawn')
        queue = ctxt.Queue()
        p = ctxt.Process(target=EngineProcess.launch, args=(queue,))
        p.start()

        pointcloud = PointCloudModelData(xyz=np.random.randn(100, 3), default_color=None)
        message = UpdateModel(model_id=0, model=pointcloud)
        queue.put(pickle.dumps(message))
        queue.put(pickle.dumps(StartEngine(), protocol=-1))
        queue.put(pickle.dumps(CloseEngine(force_close=False), protocol=-1))

        # pointcloud = PointCloudModelData(xyz=np.random.randn(100, 3), default_color=farray([[1.0, 0.0, 0.0]]))
        # message = pickle.dumps(UpdateModel(model_id=1, model=pointcloud), protocol=-1)
        # queue.put(message)
        # queue.put(pickle.dumps(StartEngine(), protocol=-1))
        queue.put(pickle.dumps(CloseProcess(force_close=True), protocol=-1))
        p.join()

        self.assertTrue(True)

    def test_window(self):
        window = OpenGLWindow()
        window.init(True)

        window.set_pointcloud(0, np.random.randn(10000, 3), point_size=1)
        window.set_pointcloud(1, -1 + 2 * np.random.rand(10000, 3), default_color=farray([[1.0, 0.0, 0.0]]),
                              point_size=10)

        window.update_camera(farray([[1.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 100.0],
                                     [0.0, 0.0, 0.0, 1.0]]))
        window.close()

        self.assertTrue(True)

    def test_cameras(self):
        engine = ExplorationEngine()
        instances = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(3, axis=0)
        instances[0, 0, 3] = -10
        instances[1, 0, 3] = 0
        instances[2, 0, 3] = 10

        engine.add_model(0, CamerasModelData(instance_model_to_world=instances, camera_size=0.5, width=1))
        engine.main_loop()
        self.assertEqual(True, True)

    def test_cameras_in_window(self):
        window = OpenGLWindow()
        window.init(True)

        window.set_pointcloud(0, np.random.randn(10000, 3), point_size=1)
        window.set_pointcloud(1, -1 + 2 * np.random.rand(10000, 3), default_color=farray([[1.0, 0.0, 0.0]]),
                              point_size=10)

        positions = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(3, axis=0)
        positions[0, 1, 3] = -10
        positions[1, 1, 3] = 0
        positions[1, 1, 3] = 10
        window.set_cameras(2, positions)

        camera_position = farray([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 100.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        window.update_camera(camera_position)

        window.set_cameras(0, camera_position.reshape(1, 4, 4), default_color=farray([[1.0, 0.0, 0.0]]))
        window.close()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
