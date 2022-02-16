from lib.Simulators.StoreSimulator import StoreSimulator
from lib.Simulators.MeterSimulator import MeterSimulator

class Simulators:

    def __init__(self):
        ss = StoreSimulator(stream_id=0)
        ms = MeterSimulator(stream_id=1)


if __name__ == '__main__':
    sim = Simulators()