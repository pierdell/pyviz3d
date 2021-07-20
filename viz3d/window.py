import multiprocessing as mp

from viz3d.engineprocess import *
from viz3d.opengl.model import *


class OpenGLWindow:
    """
    An OpenGLWindow launches an EngineProcess and communicates with it to updates its models, camera position, etc...
    """

    def __init__(self,
                 default_color: np.ndarray = np.array([[0.0, 0.6, 1.0]]),
                 engine_config: Optional[dict] = None):
        self._engine_config = engine_config
        self.default_color = default_color

        # MultiProcessing context : variables which allow one-way communication with the EngineProcess
        self._initialized = False
        self._engine_process: Optional[mp.Process] = None
        self._queue: Optional[mp.Queue] = None
        self._mp_context: Optional[mp.BaseContext] = None

        # Keep tracks of the model added / deleted to avoid useless communications

    def init(self, start_engine: bool = True):
        """
        Initialize the OpenGLWindow

        Args:
            start_engine (bool): Whether to start the engine in the listening process
        """
        if self._initialized:
            message = "The OpenGLWindow is already initialized... Close it before calling init() again."
            logging.warning(message)
            return

        self._mp_context = mp.get_context("spawn")
        self._queue = self._mp_context.Queue()
        self._engine_process = self._mp_context.Process(target=EngineProcess.launch,
                                                        args=(self._queue, self._engine_config))
        # Starts the process
        self._engine_process.start()
        logging.info("Launched a new Process for the Engine")
        self._initialized = True

        if start_engine:
            # Starts the engine in the listening process
            self.start_engine()
            logging.info("Started the Engine")

    def start_engine(self):
        """Starts the engine in the listening `EngineProcess`"""
        self._put_message(StartEngine())

    def close(self, force_close: bool = False):
        """
        Closes the OpenGLWindow

        Sends a message to the Engine process to close, and waits for it to close

        Args:
            force_close (bool): Whether to exit the main_loop are wait for the user to close the window manually
        """
        if not self._initialized:
            return

        self._put_message(CloseProcess(force_close=force_close))
        self._engine_process.join()

        self._engine_process = None
        self._mp_context = None
        self._queue.close()
        self._queue = None
        self._initialized = False

    def delete_model(self, model_id: int):
        """Deletes the model identified by `model_id`"""
        self._put_message(DeleteModel(model_id=model_id))

    def set_lines(self,
                  model_id: int,
                  lines_data: np.ndarray,
                  default_color: np.ndarray = np.zeros((1, 3), dtype=np.float32),
                  line_width: float = 1.0,
                  line_color: Optional[np.ndarray] = None):
        """
        Sets/Updates line models to be rendered in the OpenGL Window

        Args:
            model_id (int): The identifying integer of the model for the engine (will override previous models)
            lines_data (np.ndarray): The array of lines to render `(N, 2, 3)`
            default_color (np.ndarray): The default color of the lines (black by default)
            line_width (float): The width of the lines rendered
            line_color (np.ndarray): Optionally the color for each line `(N, 3)`
        """
        check_sizes(lines_data, [-1, 2, 3])
        check_sizes(default_color, [1, 3])
        if line_color is not None:
            check_sizes(line_color, [lines_data.shape[0], 3])
        model_data = LinesModelData(default_color=default_color,
                                    line_data=lines_data,
                                    line_width=line_width,
                                    line_color=line_color)
        self._put_message(UpdateModel(model_id=model_id, model=model_data))

    def set_voxels(self,
                   model_id: int,
                   points: np.ndarray,
                   voxel_size: float = 1.0,
                   default_color: np.ndarray = np.zeros((1, 3), dtype=np.float32),
                   line_width: float = 1.0):
        """
        Sets/Updates voxel models to be rendered in the OpenGL Window

        Args:
            model_id (int): The identifying integer of the model for the engine (will override previous models)
            points (np.ndarray): The array of points to render in a voxel grid `(N, 3)`
            voxel_size (float): The size of the voxels to render
            default_color (np.ndarray): The default color of the lines (black by default)
            line_width (float): The width of the lines rendered
        """
        check_sizes(points, [-1, 3])
        check_sizes(default_color, [1, 3])
        model_data = VoxelsModelData(default_color=default_color,
                                     voxel_points=points,
                                     voxel_size=voxel_size,
                                     line_width=line_width)
        self._put_message(UpdateModel(model_id, model=model_data))

    def set_ellipses(self,
                     model_id: int,
                     centers: Optional[np.ndarray] = None,
                     covariances: Optional[np.ndarray] = None,
                     colors: Optional[np.ndarray] = None,
                     default_color: np.ndarray = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)):
        """
        Sets/Updates low-polygons ellipses models to be rendered in the OpenGL Window

        Args:
            model_id (int): The identifying integer of the model for the engine (will override previous models)
            centers (np.ndarray): The centers of the ellipses to render (if None, a unique sphere will be rendered)
                                  `(N, 3)`
            covariances (np.ndarray): The symmetric matrices defining the shape of the ellipse
                                      (typically a covariance matrix) `(N, 3, 3)`
            colors (np.ndarray): The colors of the ellipses
            default_color (np.ndarray): The default color of the lines (red by default)
        """
        check_sizes(centers, [-1, 3])
        check_sizes(covariances, [-1, 3, 3])
        check_sizes(default_color, [1, 3])
        model_data = EllipsesModelData(default_color=default_color, colors=colors,
                                       means=centers, covariances=covariances)
        self._put_message(UpdateModel(model_id, model=model_data))

    def set_cameras(self,
                    model_id: int,
                    positions: np.ndarray,
                    default_color: np.ndarray = np.zeros((1, 3), dtype=np.float32),
                    line_width: float = 1.0,
                    scale: float = 1.0):
        """
        Sets/Updates camera models to be rendered in the OpenGL Window

        A camera model is a simple oriented pyramid shape indicating the position and orientation of a camera

        Args:
            model_id (int): The identifying id of the model for the engine (will override previous models)
            positions (np.ndarray): The positions of the cameras to indicate `(N, 4, 4)`
            default_color (np.ndarray): The default color of the cameras (black by default)
            line_width (float): The width of the lines of the model
            scale (float): The scale of the camera model
        """
        check_sizes(positions, [-1, 4, 4])
        check_sizes(default_color, [-1, 3])

        model_data = CamerasModelData(instance_model_to_world=positions.astype(np.float32),
                                      default_color=default_color,
                                      width=line_width,
                                      camera_size=scale)
        self._put_message(UpdateModel(model_id=model_id, model=model_data))

    def set_poses(self,
                  model_id: int,
                  positions: np.ndarray,
                  line_width: float = 1.0,
                  scale: float = 1.0):
        """
        Sets/Updates the poses model to be rendered in the OpenGL Window
        """
        check_sizes(positions, [-1, 4, 4])
        model_data = PosesModelData(instance_model_to_world=positions.astype(np.float32),
                                    width=line_width,
                                    scale=scale)
        self._put_message(UpdateModel(model_id=model_id, model=model_data))

    def set_pointcloud(self,
                       model_id: int,
                       pointcloud: np.ndarray,
                       color: Optional[np.ndarray] = None,
                       default_color: Optional[np.ndarray] = None,
                       point_size: int = 2):
        """
        Sets/Updates a point cloud to be rendered in the OpenGLWindow

        Args:
            model_id (int): The identifying id of the model for the engine (will override previous models)
            pointcloud (np.ndarray): The Point Cloud geometric data `(N, 3)`
            color (np.ndarray): The (Optional) color array describing the color of each point `(N,3)`
            default_color (np.ndarray): The (Optional) default color for the point cloud `(1, 3)`
            point_size (int): The pixel size of the points displayed by OpenGL
        """
        check_sizes(pointcloud, [-1, 3])
        if color is not None:
            check_sizes(color, [pointcloud.shape[0], 3])
        if default_color is not None:
            check_sizes(default_color, [1, 3])

        model_data = PointCloudModelData(xyz=pointcloud,
                                         color=color,
                                         default_color=default_color,
                                         point_size=point_size)

        self._put_message(UpdateModel(model_id=model_id, model=model_data))

    def update_camera(self, camera_pose: np.ndarray):
        """Updates the camera position for the view rendered in the screen"""
        check_sizes(camera_pose, [4, 4])
        self._put_message(UpdateCameraPosition(camera_position=camera_pose))

    # ------------------------------------------------------------------------------------------------------------------
    def _put_message(self, message: Message):
        """Puts the message Message in the Queue"""
        assert_debug(isinstance(message, Message))
        assert_debug(self._initialized, "Cannot Send messages, the EngineProcess is not started.")
        self._queue.put(pickle.dumps(message, protocol=-1))
