class Simulator:
    """Base class - required by PIDController."""
    
    def __init__(self, min_input, max_input):
        self.min_input = min_input
        self.max_input = max_input
    
    def set_input(self, val):
        raise NotImplementedError
    
    def get_output(self):
        raise NotImplementedError
    
    def step(self, dt):
        raise NotImplementedError
