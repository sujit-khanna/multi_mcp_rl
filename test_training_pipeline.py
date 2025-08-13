#!/usr/bin/env python3
"""
Test script to verify all components of the training pipeline
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Core training modules
        from training.scripts import train_qwen3_grpo_real_env
        print("✓ train_qwen3_grpo_real_env imported")
        
        from training.core.grpo_trainer_gradient_fix import GRPOTrainerGradientFix
        print("✓ GRPOTrainerGradientFix imported")
        
        from training.core.qwen_policy_with_value_prompting import QwenPolicyWithValuePrompting
        print("✓ QwenPolicyWithValuePrompting imported")
        
        from training.data.trajectory_collector import TrajectoryCollector
        print("✓ TrajectoryCollector imported")
        
        # Environment modules
        from environments.mcp_tool_environment import MCPToolEnvironment
        print("✓ MCPToolEnvironment imported")
        
        from environments.simple_shared_manager import SimpleSharedMCPToolManager
        print("✓ SimpleSharedMCPToolManager imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_files():
    """Test if required data files exist"""
    print("\nTesting data files...")
    
    data_files = [
        "data/inputs/train.json",
        "data/inputs/validation.json",
    ]
    
    all_exist = True
    for file_path in data_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✓ {file_path} exists ({full_path.stat().st_size:,} bytes)")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_config_files():
    """Test if required config files exist"""
    print("\nTesting config files...")
    
    config_files = [
        "training/configs/training_config_qwen3_0.6b.yaml",
        "training/configs/grpo_config_fixed.yaml",
    ]
    
    all_exist = True
    for file_path in config_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_environment_vars():
    """Test environment variables"""
    print("\nTesting environment variables...")
    
    # Check if .env file exists
    env_paths = [
        Path(".env"),
        Path("../.env"),
    ]
    
    env_found = False
    for env_path in env_paths:
        if env_path.exists():
            print(f"✓ .env file found at {env_path}")
            env_found = True
            break
    
    if not env_found:
        print("✗ .env file not found")
    
    # Check specific environment variables
    important_vars = ["DEVICE_TYPE", "PYTHONPATH"]
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"✓ {var} = {value}")
        else:
            print(f"  {var} not set (optional)")
    
    return env_found

def test_mcp_tools():
    """Test if MCP tools directory exists"""
    print("\nTesting MCP tools...")
    
    # Check relative path
    mcp_paths = [
        Path("mcp_tools/limited"),
        Path("../mcp_tools/limited"),
    ]
    
    mcp_found = False
    for mcp_path in mcp_paths:
        if mcp_path.exists():
            print(f"✓ MCP tools directory found at {mcp_path}")
            # List servers
            py_files = list(mcp_path.glob("*.py"))
            print(f"  Found {len(py_files)} Python files")
            mcp_found = True
            break
    
    if not mcp_found:
        print("✗ MCP tools directory not found")
        print("  Expected at: mcp_tools/limited or ../mcp_tools/limited")
    
    return mcp_found

def test_device():
    """Test PyTorch device availability"""
    print("\nTesting PyTorch device...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) available")
        else:
            print("✓ CPU mode")
        
        # Test device from environment
        device_type = os.environ.get('DEVICE_TYPE', 'auto')
        print(f"  DEVICE_TYPE = {device_type}")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch device check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("TRAINING PIPELINE VERIFICATION")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Files", test_data_files),
        ("Config Files", test_config_files),
        ("Environment Variables", test_environment_vars),
        ("MCP Tools", test_mcp_tools),
        ("PyTorch Device", test_device),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:25} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All tests passed! The training pipeline is ready to run.")
        print("\nTo start training, run:")
        print("  ./training/scripts/launch_real_env_cpu.sh")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before running training.")
        print("\nCommon fixes:")
        print("  1. Ensure MCP tools are in ../mcp_tools/limited/")
        print("  2. Create .env file with required API keys")
        print("  3. Install missing Python packages")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())