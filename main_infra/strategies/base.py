from abc import ABC, abstractmethod

class StrategyBase(ABC):
    def __init__(self,instrument):
        self.instrument = instrument
        if self.instrument.indicator_data is None:
            self.instrument.indicator_data = self.instrument.calculate_all_indicators()
    
    @abstractmethod
    def generate_signals(self):
        pass
