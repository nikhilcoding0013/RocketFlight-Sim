import numpy as np
from Simulator import Simulator

def air_density(altitude):
    """Calculate air density at altitude."""
    rho_0 = 1.06  # calibrated density (kg/m³)
    scale_height = 8500  # meters
    return rho_0 * np.exp(-altitude / scale_height)

class RocketAltitudeSimulator(Simulator):
    """
    Rocket simulator compatible with PIDController.
    
    Input: brake_position (0.0 = closed, 1.0 = fully open)
    Output: current altitude (meters)
    """
    
    def __init__(self, h0, v0, mass, Cd, A_base, A_max):
        """
        Initialize rocket.
        
        h0: initial altitude (m)
        v0: initial velocity (m/s)
        mass: rocket mass (kg)
        Cd: drag coefficient
        A_base: base drag area (m²)
        A_max: max drag area with brakes (m²)
        """
        super().__init__(min_input=0.0, max_input=1.0)
        
        # Rocket parameters
        self.mass = mass
        self.Cd = Cd
        self.A_base = A_base
        self.A_max = A_max
        self.g = 9.81  # m/s²
        
        # Initial conditions (for reset)
        self.h0 = h0
        self.v0 = v0
        
        # Current state
        self.h = h0
        self.v = v0
        self.brake_position = 0.0
    
    def set_input(self, brake_position):
        """
        Set airbrake position.
        
        brake_position: 0.0 (closed) to 1.0 (fully open)
        """
        self.brake_position = np.clip(brake_position, 0.0, 1.0)
    
    def get_output(self):
        """
        Get current altitude.
        
        Returns: altitude in meters
        """
        return self.h
    
    def step(self, dt):
        """
        Step simulation forward by dt seconds.
        
        dt: timestep (seconds)
        """
        # Calculate current drag area based on brake position
        A = self.A_base + (self.A_max - self.A_base) * self.brake_position
        
        # Get air density at current altitude
        rho = air_density(self.h)
        
        # Calculate drag force
        F_drag = 0.5 * rho * self.Cd * A * self.v**2
        
        # Net acceleration
        a = -self.g - F_drag / self.mass
        
        # Update velocity and altitude
        self.v = self.v + a * dt
        self.h = self.h + self.v * dt
        
        # Don't go below ground
        if self.h < 0:
            self.h = 0
            self.v = 0
    
    def get_velocity(self):
        """Get current velocity (m/s)."""
        return self.v
    
    def reset(self):
        """Reset to initial conditions."""
        self.h = self.h0
        self.v = self.v0
        self.brake_position = 0.0
