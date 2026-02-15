import numpy as np
from scipy.optimize import minimize
from RocketAltitudeSimulator import RocketAltitudeSimulator
from RocketPIDController import RocketPIDController


def simulate_with_pid(gains, rocket_params, target_apogee, dt=0.01, max_time=5.0):
    """
    Simulate rocket flight with given PID gains.
    
    Returns final apogee altitude.
    """
    kp, ki, kd = gains
    
    # Create fresh rocket
    rocket = RocketAltitudeSimulator(**rocket_params)
    
    # Create PID controller
    pid = RocketPIDController(rocket, target_apogee, kp, ki, kd, dt)
    
    # Simulate until apogee or timeout
    t = 0
    while rocket.get_velocity() > 0 and t < max_time:
        pid.step()
        t += dt
    
    return rocket.get_altitude()


def cost_function(gains, rocket_params, target_apogee):
    """
    Cost function to minimize: absolute error from target.
    """
    try:
        final_apogee = simulate_with_pid(gains, rocket_params, target_apogee)
        cost = abs(final_apogee - target_apogee)
        
        # Print progress
        kp, ki, kd = gains
        print(f"  Testing: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f} → Apogee={final_apogee:.2f}m, Error={cost:.2f}m")
        
        return cost
    except Exception as e:
        print(f"  Simulation failed: {e}")
        return 1e6  # Large penalty for failed simulations


def optimize_pid_gains(rocket_params, target_apogee):
    """
    Use scipy.optimize to find optimal PID gains.
    """
    print("=" * 70)
    print("PID Gain Optimization using Scipy")
    print("=" * 70)
    print(f"Target apogee: {target_apogee}m")
    print(f"Initial: h={rocket_params['h0']}m, v={rocket_params['v0']}m/s\n")
    
    # Initial guess
    initial_gains = [0.1, 0.01, 0.05]  # [Kp, Ki, Kd]
    
    print("Starting optimization...\n")
    
    # Run optimization
    result = minimize(
        cost_function,
        x0=initial_gains,
        args=(rocket_params, target_apogee),
        method='Nelder-Mead',  # Doesn't require gradients!
        options={
            'maxiter': 100,
            'xatol': 0.001,  # Tolerance for convergence
            'disp': True
        }
    )
    
    optimal_kp, optimal_ki, optimal_kd = result.x
    final_error = result.fun
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Optimal PID Gains:")
    print(f"  Kp = {optimal_kp:.6f}")
    print(f"  Ki = {optimal_ki:.6f}")
    print(f"  Kd = {optimal_kd:.6f}")
    print(f"\nFinal error: {final_error:.3f}m")
    print(f"Converged: {result.success}")
    print(f"Iterations: {result.nit}")
    
    return optimal_kp, optimal_ki, optimal_kd


def simulate_and_plot(rocket_params, target_apogee, kp, ki, kd):
    """
    Simulate with optimized gains and plot trajectory.
    """
    import matplotlib.pyplot as plt
    
    rocket = RocketAltitudeSimulator(**rocket_params)
    pid = RocketPIDController(rocket, target_apogee, kp, ki, kd, dt=0.01)
    
    times = []
    altitudes = []
    velocities = []
    brake_positions = []
    predictions = []
    
    t = 0
    while rocket.get_velocity() > 0 and t < 10:
        predicted_apogee, error, brake_pos = pid.step()
        
        times.append(t)
        altitudes.append(rocket.get_altitude())
        velocities.append(rocket.get_velocity())
        brake_positions.append(brake_pos)
        predictions.append(predicted_apogee)
        
        t += 0.01
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Altitude
    ax1 = axes[0]
    ax1.plot(times, altitudes, 'b-', linewidth=2, label='Actual altitude')
    ax1.plot(times, predictions, 'r--', linewidth=2, alpha=0.7, label='Predicted apogee')
    ax1.axhline(y=target_apogee, color='g', linestyle='--', linewidth=2, label=f'Target ({target_apogee}m)')
    ax1.set_ylabel('Altitude (m)', fontsize=12)
    ax1.set_title(f'PID Control (Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Velocity
    ax2 = axes[1]
    ax2.plot(times, velocities, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Brake position
    ax3 = axes[2]
    ax3.plot(times, brake_positions, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Brake Position (0-1)', fontsize=12)
    ax3.set_ylim([-0.1, 1.1])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal apogee: {rocket.get_altitude():.2f}m")
    print(f"Error from target: {rocket.get_altitude() - target_apogee:+.2f}m")
