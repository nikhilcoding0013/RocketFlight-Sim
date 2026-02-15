    target_apogee = 228.6  # meters
    
    print("=" * 60)
    print("PID Auto-Tuning for Rocket Airbrakes")
    print("=" * 60)
    print(f"Target apogee: {target_apogee}m")
    print(f"Initial conditions: h={rocket_params['h0']}m, v={rocket_params['v0']}m/s")
    print("\nStarting PID optimization...")
    
    # Create rocket simulator
    rocket = RocketAltitudeSimulator(**rocket_params)
    
    # Create PID tuner
    tuner = PIDTuner(
        sim=rocket,
        setpoint=target_apogee,
        t0=0.0,
        t1=5.0,  # simulate 5 seconds (should reach apogee)
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
    print("RESULTS")
    print("=" * 60)
    print(f"Optimized PID gains:")
    print(f"  Kp = {kp:.6f}")
    print(f"  Ki = {ki:.6f}")
    print(f"  Kd = {kd:.6f}")
