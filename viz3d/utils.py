import numpy as np


def assert_debug(condition: bool, message: str = ""):
    """
    Debug Friendly assertion

    Allows to put a breakpoint, and catch any assertion error in debug
    """
    if not condition:
        print(f"[ERROR][ASSERTION]{message}")
        raise AssertionError(message)


def check_sizes(tensor: np.ndarray, sizes: list):
    """
    Checks the size of a tensor along all its dimensions, against a list of sizes

    The tensor must have the same number of dimensions as the list sizes
    For each dimension, the tensor must have the same size as the corresponding entry in the list
    A size of -1 in the list matches all sizes

    Any Failure raises an AssertionError
    """
    tensor_shape = list(tensor.shape)
    assert_debug(len(tensor_shape) == len(sizes),
                 f"[BAD TENSOR SHAPE] Wrong tensor shape got {tensor_shape} expected {sizes}")
    for i in range(len(sizes)):
        assert_debug(sizes[i] == -1 or sizes[i] == tensor_shape[i],
                     f"[BAD TENSOR SHAPE] Wrong tensor shape got {tensor_shape} expected {sizes}")
