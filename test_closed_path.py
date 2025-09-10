#!/usr/bin/env python3
"""
Simple test script to verify closed path functionality
"""

import numpy as np
import torch
import sys
import os

# Add current directory to path
sys.path.append('.')

# Import the main file
exec(open('11+9+10_jiasu 无课程学习.py').read())

def test_closed_path_tracking():
    """Test that closed path progress tracking works correctly"""
    
    # Create a simple square closed path
    square_path = [
        [0.0, 0.0],    # Start point
        [1.0, 0.0],    # Right
        [1.0, 1.0],    # Up
        [0.0, 1.0],    # Left  
        [0.0, 0.0]     # Back to start (closed)
    ]
    
    print("Testing closed path progress tracking...")
    print(f"Path: {square_path}")
    
    # Create environment with the closed path
    device = torch.device('cpu')
    env = Env(
        device=device,
        epsilon=0.1,
        interpolation_period=0.01,
        MAX_VEL=1.0,
        MAX_ACC=2.0,
        MAX_JERK=5.0,
        MAX_ANG_VEL=1.0,
        MAX_ANG_ACC=2.0,
        MAX_ANG_JERK=5.0,
        Pm=square_path,
        max_steps=1000
    )
    
    print(f"Path is closed: {env.closed}")
    print(f"Segment count: {env.segment_count}")
    print(f"Total path length: {env.cache['total_path_length']}")
    
    # Test initial state
    initial_state = env.reset()
    print(f"\nInitial state - Progress: {env.local_progress:.3f}, Cumulative: {env.cumulative_progress:.3f}, Laps: {env.completed_laps}")
    
    # Test a few manual positions along the path
    test_positions = [
        [0.5, 0.0],    # Halfway along first segment
        [1.0, 0.5],    # Halfway along second segment
        [0.5, 1.0],    # Halfway along third segment  
        [0.0, 0.5],    # Halfway along fourth segment
        [0.1, 0.0],    # Near the start (should detect lap completion)
    ]
    
    print("\nTesting progress at various positions:")
    for i, pos in enumerate(test_positions):
        env.current_position = np.array(pos)
        progress = env._calculate_path_progress(env.current_position)
        print(f"Position {pos} -> Progress: {progress:.3f}, Local: {env.local_progress:.3f}, "
              f"Cumulative: {env.cumulative_progress:.3f}, Laps: {env.completed_laps}, "
              f"Completed: {env.path_completed}")
    
    # Test is_done method
    print(f"\nPath completion check: {env.is_done()}")
    
    return env

def test_non_closed_path_compatibility():
    """Test that non-closed paths still work as before"""
    
    # Create a simple linear path (not closed)
    linear_path = [
        [0.0, 0.0],
        [1.0, 0.0], 
        [2.0, 0.0],
        [3.0, 0.0]
    ]
    
    print("\n" + "="*50)
    print("Testing non-closed path compatibility...")
    print(f"Path: {linear_path}")
    
    device = torch.device('cpu')
    env = Env(
        device=device,
        epsilon=0.1,
        interpolation_period=0.01,
        MAX_VEL=1.0,
        MAX_ACC=2.0,
        MAX_JERK=5.0,
        MAX_ANG_VEL=1.0,
        MAX_ANG_ACC=2.0,
        MAX_ANG_JERK=5.0,
        Pm=linear_path,
        max_steps=1000
    )
    
    print(f"Path is closed: {env.closed}")
    
    # Test initial state
    initial_state = env.reset()
    print(f"\nInitial progress: {env.local_progress:.3f}")
    
    # Test progress at end
    env.current_position = np.array([3.0, 0.0])
    progress = env._calculate_path_progress(env.current_position)
    print(f"End position progress: {progress:.3f}")
    print(f"Should be done: {env.is_done()}")
    
    return env

if __name__ == "__main__":
    try:
        print("Running closed path tests...")
        closed_env = test_closed_path_tracking()
        
        non_closed_env = test_non_closed_path_compatibility()
        
        print("\n" + "="*50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()