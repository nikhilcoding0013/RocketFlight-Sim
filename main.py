from Optimization import optimize_pid_gains, simulate_and_plot


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
    
    target_apogee = 228.6
    
    # Optimize PID gains
    kp, ki, kd = optimize_pid_gains(rocket_params, target_apogee)
    
    # Simulate and visualize with optimal gains
    print("\nSimulating with optimal gains...")
    simulate_and_plot(rocket_params, target_apogee, kp, ki, kd)


if __name__ == "__main__":
    main()
