import unittest

from common import get_logger
from tensorflow.python.client import device_lib

class TestGPUAccess(unittest.TestCase):

    def test_gpu(self):
        logger = get_logger("RL.Test.GPUAccess")
        local_devices = device_lib.list_local_devices()
        logger.info("Available Devices: %s", str(local_devices))
        self.assertTrue(any(['GPU' in str(d) for d in local_devices]))