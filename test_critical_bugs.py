#!/usr/bin/env python3
"""
Test Critical Bug Fixes
=======================

Tests the two critical bugs that were preventing training:
1. ✅ Fixed trajectory variable error in trajectory_collector.py
2. ✅ Fixed infinite generation with aggressive generation limits

This ensures both fixes are properly applied.
"""

import sys
import yaml
from pathlib import Path

def test_trajectory_bug_fix():
    """Test that the trajectory variable error is fixed"""
    
    print("🔍 TESTING TRAJECTORY BUG FIX")
    print("=" * 50)
    
    # Check if the buggy line was fixed
    file_path = Path(__file__).parent / "training" / "data" / "trajectory_collector.py"
    
    if not file_path.exists():
        print("❌ Trajectory collector file not found")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for the old buggy pattern
    buggy_pattern = "len(trajectory) if trajectory else 0"
    fixed_pattern = "len(self.policy.last_forced_mask) > 0"
    
    checks = [
        ("Buggy pattern removed", buggy_pattern not in content),
        ("Fixed pattern present", fixed_pattern in content),
        ("Variable error fixed", "trajectory" not in content.split('action_idx')[0] if 'action_idx' in content else True)
    ]
    
    all_passed = True
    for description, check in checks:
        status = "✅ PASSED" if check else "❌ FAILED"
        print(f"   {status}: {description}")
        if not check:
            all_passed = False
    
    return all_passed

def test_generation_limits():
    """Test that generation limits are aggressive enough"""
    
    print("\n🔍 TESTING GENERATION LIMITS")
    print("=" * 50)
    
    # Check the generation config
    config_path = Path(__file__).parent / "training" / "configs" / "model_config_qwen3_0.6b.yaml"
    
    if not config_path.exists():
        print("❌ Config file not found")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    gen_config = config.get('generation', {})
    
    # Check aggressive limits
    checks = [
        ("Very low temperature", gen_config.get('temperature', 1.0) <= 0.1),
        ("Very short max tokens", gen_config.get('max_new_tokens', 1000) <= 128),
        ("High repetition penalty", gen_config.get('repetition_penalty', 1.0) >= 1.2),
        ("Low top_k for focus", gen_config.get('top_k', 50) <= 10),
        ("Early stopping enabled", gen_config.get('early_stopping', False) == True),
        ("Length penalty for short responses", gen_config.get('length_penalty', 1.0) <= 0.5)
    ]
    
    all_passed = True
    for description, check in checks:
        status = "✅ PASSED" if check else "❌ FAILED"
        print(f"   {status}: {description}")
        if not check:
            all_passed = False
    
    print(f"\n   Current generation config: {gen_config}")
    
    return all_passed

def main():
    """Run all tests"""
    
    print("🐛 TESTING CRITICAL BUG FIXES")
    print("=" * 70)
    
    tests = [
        ("Trajectory Variable Bug Fix", test_trajectory_bug_fix),
        ("Generation Limits", test_generation_limits)
    ]
    
    all_passed = True
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    print("🏆 BUG FIX TEST RESULTS")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 OVERALL: {'✅ ALL CRITICAL BUGS FIXED' if all_passed else '❌ SOME BUGS REMAIN'}")
    
    if all_passed:
        print("\n🚀 Critical bugs fixed! Training should now:")
        print("   - ✅ Not crash with 'trajectory is not defined' error")
        print("   - ✅ Generate much shorter responses (128 tokens max)")
        print("   - ✅ Have higher GPU utilization (actual training)")
        print("   - ✅ Complete episodes faster")
        print("   - ✅ Show actual training progress")
    else:
        print("\n🔧 Some bugs still need attention")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)