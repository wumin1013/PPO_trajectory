#!/usr/bin/env python3
"""
Simple test for just the Env class closed path functionality
"""
import numpy as np
import torch
import math
from numba import jit
from rtree import index

# Copy just the needed parts from the main file
@jit(nopython=True)
def apply_kinematic_constraints(prev_vel, prev_acc, prev_ang_vel, prev_ang_acc,
                               vel_action, ang_vel_action, dt,
                               MAX_VEL, MAX_ACC, MAX_JERK,
                               MAX_ANG_VEL, MAX_ANG_ACC, MAX_ANG_JERK):
    # Simplified version for testing
    return (vel_action, 0.0, 0.0, ang_vel_action, 0.0, 0.0)

# Test just the core methods we modified
def test_progress_calculation():
    """Test progress calculation directly"""
    
    # Create a simple square path
    square_path = [
        [0.0, 0.0],    # Start point
        [1.0, 0.0],    # Right
        [1.0, 1.0],    # Up
        [0.0, 1.0],    # Left  
        [0.0, 0.0]     # Back to start (closed)
    ]
    
    print("Testing path progress calculation...")
    
    # Mock a minimal Env class with just what we need
    class MinimalEnv:
        def __init__(self, path_points):
            self.Pm = [np.array(p) for p in path_points]
            self.closed = len(path_points) > 2 and np.allclose(path_points[0], path_points[-1], atol=1e-6)
            print(f"Path is closed: {self.closed}")
            
            # Initialize progress tracking
            self.completed_laps = 0
            self.cumulative_progress = 0.0
            self.local_progress = 0.0
            self.path_completed = False
            self.last_progress = 0.0
            
            # Calculate total length
            total_length = 0.0
            for i in range(len(self.Pm) - 1):
                total_length += np.linalg.norm(self.Pm[i+1] - self.Pm[i])
            if self.closed and len(self.Pm) > 2:
                total_length += np.linalg.norm(self.Pm[0] - self.Pm[-1])
            
            self.cache = {'total_path_length': total_length}
            print(f"Total path length: {total_length}")
            
        def _find_containing_segment(self, pt):
            """Find which segment contains the point"""
            min_dist = float('inf')
            best_segment = -1
            
            n = len(self.Pm)
            segments = n if self.closed else n-1
            
            for i in range(segments):
                j = (i + 1) % n
                p1, p2 = self.Pm[i], self.Pm[j]
                
                # Project point to segment
                seg_vec = p2 - p1
                pt_vec = pt - p1
                
                if np.dot(seg_vec, seg_vec) < 1e-6:
                    continue
                    
                t = np.clip(np.dot(pt_vec, seg_vec) / np.dot(seg_vec, seg_vec), 0, 1)
                projection = p1 + t * seg_vec
                dist = np.linalg.norm(pt - projection)
                
                if dist < min_dist:
                    min_dist = dist
                    best_segment = i
            
            return best_segment if min_dist < 0.5 else -1  # tolerance
        
        def _calculate_path_progress(self, pt):
            """Our modified progress calculation"""
            total_length = self.cache['total_path_length'] or 1.0
            
            # Find current segment
            segment_index = self._find_containing_segment(pt)
            if segment_index < 0:
                return 0.0
                
            current_dist = 0.0
            
            # Add lengths of previous segments
            for i in range(segment_index):
                j = (i + 1) % len(self.Pm)
                current_dist += np.linalg.norm(self.Pm[j] - self.Pm[i])
            
            # Add progress within current segment
            if self.closed and segment_index == len(self.Pm) - 1:
                p1, p2 = self.Pm[-1], self.Pm[0]
            else:
                p1, p2 = self.Pm[segment_index], self.Pm[segment_index + 1]
                
            seg_vec = p2 - p1
            pt_vec = pt - p1
            seg_length = np.linalg.norm(seg_vec)
            
            if seg_length > 1e-6:
                t = np.clip(np.dot(pt_vec, seg_vec) / (seg_length ** 2), 0, 1)
                current_dist += t * seg_length
            
            progress = current_dist / total_length
            
            if self.closed:
                # Our new closed path logic
                self.local_progress = progress
                
                # Detect lap completion
                if self.last_progress > 0.95 and progress < 0.1:
                    self.completed_laps += 1
                    self.cumulative_progress += 1.0
                    
                    if self.completed_laps >= 1:
                        self.path_completed = True
                
                current_cumulative = self.completed_laps + progress
                self.cumulative_progress = current_cumulative
                
                if self.path_completed:
                    return min(current_cumulative, 1.001)
                else:
                    return min(progress, 1.0)
            else:
                return progress
    
    # Test the closed path
    env = MinimalEnv(square_path)
    
    # Test positions around the square - simulate completing a full lap
    test_positions = [
        ([0.0, 0.0], "Start"),
        ([0.5, 0.0], "First segment middle"),
        ([1.0, 0.0], "First corner"), 
        ([1.0, 0.5], "Second segment middle"),
        ([1.0, 1.0], "Second corner"),
        ([0.5, 1.0], "Third segment middle"),
        ([0.0, 1.0], "Third corner"),
        ([0.0, 0.5], "Fourth segment middle"),
        ([0.0, 0.1], "Near start (fourth segment end)"),
        ([0.0, 0.0], "Back to start - should complete lap"),
        ([0.1, 0.0], "Just past start - should be in second lap"),
    ]
    
    print("\nTesting progress at different positions:")
    for pos, desc in test_positions:
        pt = np.array(pos)
        progress = env._calculate_path_progress(pt)
        print(f"{desc:25} {pos} -> Progress: {progress:.3f}, Local: {env.local_progress:.3f}, "
              f"Laps: {env.completed_laps}, Completed: {env.path_completed}")
        
        # Update last_progress for next iteration
        env.last_progress = env.local_progress
    
    print(f"\nFinal state: Completed laps: {env.completed_laps}, Path completed: {env.path_completed}")
    return env

if __name__ == "__main__":
    try:
        test_progress_calculation()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()