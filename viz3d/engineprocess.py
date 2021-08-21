from abc import ABC
from multiprocessing.queues import Queue
from threading import Thread
import pickle
import logging
import numpy as np

from dataclasses import dataclass

from viz3d.opengl.pygame_engine import ExplorationEngine, Optional, assert_debug, ModelData, Model


# ----------------------------------------------------------------------------------------------------------------------
class Message(ABC):
    """An abstract message DataClass, all messages expected by the process must be of this instance"""


@dataclass
class CloseProcess(Message):
    """A Message for stopping the main loop"""
    force_close: bool = False  # Whether to force close the engine, are wait for the user to close the window manually


class StartEngine(Message):
    """A Message for starting the engine main loop"""


@dataclass
class CloseEngine(Message):
    """A Message for closing the engine main loop"""
    force_close: bool = False  # Whether to force close the engine, are wait for the user to close the window manually


@dataclass
class UpdateModel(Message):
    """A Message to add or update a `Model` to the engine"""
    __slots__ = "model_id", "model"
    model_id: int
    model: ModelData


@dataclass
class UpdateCameraPosition(Message):
    """A Message to update the camera position of the view rendered in the screen"""
    __slots__ = "camera_position"
    camera_position: np.ndarray


@dataclass
class DeleteModel(Message):
    """A message to delete a `Model` from the engine"""
    __slots__ = "model_id"
    model_id: int


# ----------------------------------------------------------------------------------------------------------------------
class EngineProcess:
    """
    A Class which manages an engine, and reads other processes messages to modify the engine state's
    """

    def __init__(self, queue: Queue, engine_config: dict = None):
        if engine_config is None:
            engine_config = {}

        self._window_thread: Optional[Thread] = None  # Runs the engine in a separate thread
        self._queue = queue  # The queue is the entry point of other processes messages
        self._engine = ExplorationEngine(**engine_config)

        self._break_loop: bool = False  # Whether to break the main loop or not

    def start(self):
        """
        Starts the engine, and starts the main loop to read messages
        """

        # Main loop
        while True:
            if self._break_loop:
                break

            message_dump = self._queue.get()
            message: Message = pickle.loads(message_dump)
            logging.debug(f"Received message : {type(message)} (id={id(message)})")

            self.handle_message(message)

    def close(self, force_close: bool = False):
        """
        Closes the engine and the message listening loop
        """
        self._break_loop = True
        if force_close:
            self._engine.force_close()
        if self._window_thread is not None:
            self._window_thread.join()
            self._window_thread = None

    @staticmethod
    def launch(queue: Queue, engine_config: dict = None):
        """
        Launches the process, which will stop when the corresponding message is received

        Usage:
           p = Process(target=EngineProcess.launch, args=(queue,))
           p.start()
           p.join()

        TODO : Set Logging in temp file ?
        """
        process = EngineProcess(queue, engine_config)
        process.start()

    # ------------------------------------------------------------------------------------------------------------------
    # MESSAGE HANDLING

    def handle_message(self, message: Message):
        """Handles messages received by other processes"""
        assert_debug(isinstance(message, Message))
        if isinstance(message, UpdateModel):
            self.__update_model(message.model_id, message.model)
        elif isinstance(message, UpdateCameraPosition):
            self.__update_camera(message.camera_position)
        elif isinstance(message, DeleteModel):
            self.__delete_model(message.model_id)
        elif isinstance(message, StartEngine):
            self.__start_engine()
        elif isinstance(message, CloseEngine):
            self.__close_engine(message.force_close)
        elif isinstance(message, CloseProcess):
            self.__close_process(message)
        else:
            raise NotImplementedError("")

    def __start_engine(self):
        if self._window_thread is None:
            self._window_thread = Thread(target=lambda: self._engine.main_loop())
            self._window_thread.start()
        else:
            logging.warning("Engine is already started.")

    def __close_engine(self, force_close: bool):
        if self._window_thread is None:
            logging.warning("The Engine is not running, cannot close it.")
            return

        if force_close:
            # Force the engine to stop its main loop
            self._engine.force_close()

        # Wait for the engine thread to finish its main loop
        self._window_thread.join()
        self._window_thread = None

    def __close_process(self, message: CloseProcess):
        if self._window_thread is not None:
            self.__close_engine(message.force_close)

        self._break_loop = True

    def __update_model(self, model_id: int, model_data: ModelData):
        if model_id not in self._engine.models:
            self._engine.add_model(model_id, model_data)
        else:
            current_model: Model = self._engine.models[model_id]
            current_model.model_data = model_data
            self._engine.update_model(model_id)

    def __delete_model(self, model_id: int):
        self._engine.delete_model(model_id)

    def __update_camera(self, camera_position: np.ndarray):
        self._engine.update_camera(camera_position)
