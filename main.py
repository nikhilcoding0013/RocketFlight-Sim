from RocketAltitudeSimulator import RocketAltitudeSimulator
from PIDController import PIDController
from PIDTuner import PIDTuner
import importlib
import RocketAltitudeSimulator as ras
importlib.reload(ras)


def main():
    rocket_params = {
        'h0': 76.25,
        'v0': 70.25,
        'mass': 0.595,
        'Cd': 0.68,
        'A_base': 0.00452369,
        'A_max': 0.00524229
    }
    
    target_apogee = 228.6
    
    print("=" * 60)
    print("PID Auto-Tuning for Rocket Airbrakes")
    print("=" * 60)
    print(f"Target apogee: {target_apogee}m")
    print(f"Initial: h={rocket_params['h0']}m, v={rocket_params['v0']}m/s")
    print("\nStarting PID optimization...")
    print("(Using NEGATIVE gains for inverted control)")
    
    rocket = RocketAltitudeSimulator(**rocket_params)
    
    tuner = PIDTuner(
        sim=rocket,
        setpoint=target_apogee,
        t0=0.0,
        t1=5.0,
        dt=0.01,
        kp_init=-0.1,      # ← NEGATIVE (inverted control)
        ki_init=-0.01,     # ← NEGATIVE
        kd_init=-0.1,      # ← NEGATIVE
        output_limits=(0.0, 1.0)
    )
    
    kp, ki, kd = tuner.optimize(
        max_epochs=100,
        gamma=0.01,
        delta=0.01,
        tol=1e-4,
        verbose=True,
        clamp_params=(-10.0, 10.0)  # ← Allow negative values
    )
    
    print("\n" + "=" * 60)
    print("OPTIMIZED PID GAINS")
    print("=" * 60)
    print(f"Kp = {kp:.6f}")
    print(f"Ki = {ki:.6f}")
    print(f"Kd = {kd:.6f}")
    print("\nNote: Negative gains are correct for inverted control!")
    print("(Predicted too low → close brakes → decrease output)")


if __name__ == "__main__":
    main()
