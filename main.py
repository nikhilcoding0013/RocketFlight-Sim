from RocketAltitudeSimulator import RocketAltitudeSimulator
from PIDController import PIDController
from PIDTuner import PIDTuner


def main():
    # Create rocket
    rocket = RocketAltitudeSimulator(
        h0=76.02,
        v0=69.95,
        mass=0.595,
        Cd=0.68,
        A_base=0.00452369,
        A_max=0.00524229
    )
    
    # TODO: Add PID tuning and comparison
    print("Rocket simulator created!")
    print(f"Initial altitude: {rocket.get_output():.2f}m")


if __name__ == "__main__":
    main()
