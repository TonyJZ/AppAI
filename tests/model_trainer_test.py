import unittest
from src.model_trainer import model_trainer


if __name__ == '__main__':
    debugInfo = {}
    debugInfo['solution_type'] = 0
    debugInfo['meter_id'] = 19

    model_trainer.model_trainer_implementation(debugInfo)