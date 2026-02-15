import numpy as np
if __name__ == "__main__":
    # Create rocket
    rocket = RocketAltitudeSimulator(
        h0=76.02,           # initial altitude
        v0=69.95,           # initial velocity
        mass=0.595,         # kg
        Cd=0.68,            # drag coefficient
        A_base=0.00452369,  # base area
        A_max=0.00524229    # max area with brakes
    )
    
    # Simulate for 1 second with no brakes
    dt = 0.01
    for i in range(100):
        rocket.set_input(0.0)  # brakes closed
        rocket.step(dt)
        if i % 10 == 0:
            print(f"t={i*dt:.2f}s: h={rocket.get_output():.2f}m, v={rocket.get_velocity():.2f}m/s")
