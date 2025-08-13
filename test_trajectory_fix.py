#!/usr/bin/env python3
"""Test script to verify the trajectory fix works correctly"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from training.core.grpo_trainer import Trajectory

def test_trajectory_to_device():
    """Test the to_device method with different log_probs formats"""
    
    print("Testing Trajectory.to_device() method...")
    
    # Test 1: log_probs as list of tensors
    print("\n1. Testing with log_probs as list of tensors...")
    device = torch.device("cpu")
    log_probs_list = [torch.tensor(-1.0), torch.tensor(-2.0), torch.tensor(-3.0)]
    
    traj1 = Trajectory(
        task_id="test1",
        states=[[], [], []],
        actions=["a1", "a2", "a3"],
        rewards=[0.1, 0.2, 0.3],
        dones=[False, False, True]
    )
    traj1.log_probs = log_probs_list
    
    # Move to device
    traj1_moved = traj1.to_device(device)
    assert isinstance(traj1_moved.log_probs, torch.Tensor), "log_probs should be tensor after to_device"
    assert traj1_moved.log_probs.shape[0] == 3, "log_probs should have 3 elements"
    print("✓ List of tensors handled correctly")
    
    # Test 2: log_probs as tensor
    print("\n2. Testing with log_probs as tensor...")
    log_probs_tensor = torch.tensor([-1.0, -2.0, -3.0])
    
    traj2 = Trajectory(
        task_id="test2",
        states=[[], [], []],
        actions=["a1", "a2", "a3"],
        rewards=[0.1, 0.2, 0.3],
        dones=[False, False, True]
    )
    traj2.log_probs = log_probs_tensor
    
    traj2_moved = traj2.to_device(device)
    assert isinstance(traj2_moved.log_probs, torch.Tensor), "log_probs should remain tensor"
    assert traj2_moved.log_probs.shape[0] == 3, "log_probs should have 3 elements"
    print("✓ Tensor handled correctly")
    
    # Test 3: log_probs as None
    print("\n3. Testing with log_probs as None...")
    traj3 = Trajectory(
        task_id="test3",
        states=[[], [], []],
        actions=["a1", "a2", "a3"],
        rewards=[0.1, 0.2, 0.3],
        dones=[False, False, True]
    )
    
    traj3_moved = traj3.to_device(device)
    assert traj3_moved.log_probs is None, "log_probs should remain None"
    print("✓ None handled correctly")
    
    # Test 4: With forced_mask
    print("\n4. Testing with forced_mask...")
    traj4 = Trajectory(
        task_id="test4",
        states=[[], [], []],
        actions=["a1", "a2", "a3"],
        rewards=[0.1, 0.2, 0.3],
        dones=[False, False, True]
    )
    traj4.forced_mask = torch.tensor([True, False, False], dtype=torch.bool)
    
    traj4_moved = traj4.to_device(device)
    assert hasattr(traj4_moved, 'forced_mask'), "forced_mask should be preserved"
    assert isinstance(traj4_moved.forced_mask, torch.Tensor), "forced_mask should be tensor"
    print("✓ forced_mask handled correctly")
    
    print("\n✅ All tests passed! The trajectory fix is working correctly.")
    return True

if __name__ == "__main__":
    test_trajectory_to_device()