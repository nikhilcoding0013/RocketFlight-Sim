from RocketAltitudeSimulator import RocketAltitudeSimulator
from PIDController import PIDController
from PIDTuner import PIDTuner


def main():
    # Rocket parameters
    rocket_params = {
        'h0': 76.25,
        'v0': 70.25,
        'mass': 0.595,
        'Cd': 0.65,
        'A_base': 0.00452369,
        'A_max': 0.00524229
    }
    
    target_apogee = 228.6  # meters
    
    print("=" * 60)
    print("PID Auto-Tuning for Rocket Airbrakes")
    print("=" * 60)
    print(f"Target apogee: {target_apogee}m")
    print(f"Initial: h={rocket_params['h0']}m, v={rocket_params['v0']}m/s")
    print("\nStarting PID optimization...")
    print("(This may take 1-2 minutes)\n")
    
    # Create rocket simulator (now predicts apogee!)
    rocket = RocketAltitudeSimulator(**rocket_params)
    
    # Create PID tuner
    tuner = PIDTuner(
        sim=rocket,
        setpoint=target_apogee,
        t0=0.0,
        t1=3.0,  # simulate 3 seconds (enough to reach apogee)
        dt=0.01,
        kp_init=0.01,
        ki_init=0.001,
        kd_init=0.01,
        output_limits=(0.0, 1.0)  # brake position 0-1
    )
    
    # Run optimization
    kp, ki, kd = tuner.optimize(
        max_epochs=100,
        gamma=0.001,  # learning rate
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("OPTIMIZED PID GAINS")
    print("=" * 60)
    print(f"Kp = {kp:.6f}")
    print(f"Ki = {ki:.6f}")
    print(f"Kd = {kd:.6f}")
    print("\nThese values minimize error between predicted and target apogee!")


if __name__ == "__main__":
    main()
