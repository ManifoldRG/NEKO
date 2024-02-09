from abc import ABC

# import logger
import logging
logger = logging.getLogger(__name__)
# Example of use logger.debug(f'foobar')

class Task(ABC):
    def sample_batch(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()
