#!/usr/bin/env python3
"""
Quick Memory Test - Lightweight version without model loading

This script tests memory monitoring and configuration without loading actual models
to avoid the bitsandbytes dependency issue.
"""

import sys
import json
import tempfile
from pathlib import Path
import time
import logging

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_memory_functions():
    """Test memory monitoring functions without model loading."""
    
    def get_memory_usage():
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.mps.current_allocated_memory() / (1024**3)
            else:
                import psutil
                return psutil.Process().memory_info().rss / (1024**3)
        except Exception:
            return 0.0
    
    initial_memory = get_memory_usage()
    
    # Allocate some tensors to test memory monitoring
    device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
    
    test_tensors = []
    for i in range(5):
        tensor = torch.randn(100, 100).to(device)
        test_tensors.append(tensor)
    
    peak_memory = get_memory_usage()
    
    # Clean up
    del test_tensors
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    
    final_memory = get_memory_usage()
    
    return {
        "device": str(device),
        "initial_memory_gb": initial_memory,
        "peak_memory_gb": peak_memory,
        "final_memory_gb": final_memory,
        "memory_allocated_gb": peak_memory - initial_memory,
        "memory_freed_gb": peak_memory - final_memory,
        "success": True
    }


def test_config_creation():
    """Test configuration file creation and parsing."""
    
    # Test configuration data
    test_config = {
        "model_name": "test_model",
        "max_length": 1024,
        "quantization": {
            "load_in_4bit": False,  # Disabled for testing
            "bnb_4bit_compute_dtype": "float32"
        },
        "lora_mode": {
            "enabled": True,
            "r": 8,
            "alpha": 16
        }
    }
    
    try:
        # Test YAML serialization
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            
            with open(config_file, 'w') as f:
                yaml.dump(test_config, f)
            
            # Test YAML deserialization
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config == test_config, "Config not preserved"
            
            return {
                "config_serialization": True,
                "config_file_size": config_file.stat().st_size,
                "success": True
            }
    
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


def get_system_info():
    """Get comprehensive system information."""
    
    import platform
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device_info": {}
    }
    
    # Test device availability
    if torch.cuda.is_available():
        system_info["device_info"]["cuda"] = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    else:
        system_info["device_info"]["cuda"] = {"available": False}
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        system_info["device_info"]["mps"] = {
            "available": True,
            "device_name": "Apple Silicon GPU"
        }
    else:
        system_info["device_info"]["mps"] = {"available": False}
    
    # System memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        system_info["memory"] = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_percent": memory.percent
        }
    except ImportError:
        system_info["memory"] = {"error": "psutil not available"}
    
    return system_info


def run_quick_memory_test():
    """Run quick memory test suite."""
    
    logger.info("Starting quick memory test...")
    start_time = time.time()
    
    results = {
        "test_suite": "Quick Memory Test",
        "timestamp": time.time(),
        "system_info": get_system_info(),
        "tests": {}
    }
    
    # Test 1: Memory monitoring
    logger.info("Testing memory monitoring...")
    try:
        memory_test = test_memory_functions()
        results["tests"]["memory_monitoring"] = memory_test
        logger.info(f"✅ Memory monitoring: {memory_test['memory_allocated_gb']:.3f}GB allocated")
    except Exception as e:
        results["tests"]["memory_monitoring"] = {"success": False, "error": str(e)}
        logger.error(f"❌ Memory monitoring failed: {e}")
    
    # Test 2: Configuration handling
    logger.info("Testing configuration handling...")
    try:
        config_test = test_config_creation()
        results["tests"]["configuration"] = config_test
        logger.info("✅ Configuration handling working")
    except Exception as e:
        results["tests"]["configuration"] = {"success": False, "error": str(e)}
        logger.error(f"❌ Configuration handling failed: {e}")
    
    # Test 3: Device detection
    logger.info("Testing device detection...")
    try:
        device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
        test_tensor = torch.randn(10, 10).to(device)
        
        results["tests"]["device_detection"] = {
            "success": True,
            "device": str(device),
            "tensor_device": str(test_tensor.device),
            "device_match": str(test_tensor.device) == str(device)
        }
        logger.info(f"✅ Device detection: {device}")
    except Exception as e:
        results["tests"]["device_detection"] = {"success": False, "error": str(e)}
        logger.error(f"❌ Device detection failed: {e}")
    
    # Summary
    successful_tests = sum(1 for test in results["tests"].values() if test.get("success", False))
    total_tests = len(results["tests"])
    
    results["summary"] = {
        "duration_seconds": time.time() - start_time,
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
        "overall_success": successful_tests == total_tests
    }
    
    return results


def main():
    """Main entry point."""
    
    print("🔍 Quick Memory Test for GRPO Training")
    print("="*50)
    
    try:
        results = run_quick_memory_test()
        
        # Save results
        output_file = Path(__file__).parent / "quick_memory_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\nTest Duration: {results['summary']['duration_seconds']:.2f} seconds")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Successful Tests: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
        
        print("\nTest Results:")
        for test_name, test_result in results["tests"].items():
            status = "✅" if test_result.get("success", False) else "❌"
            print(f"  {status} {test_name}")
        
        print(f"\nResults saved to: {output_file}")
        
        if results["summary"]["overall_success"]:
            print("🎉 All quick memory tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed.")
            return 1
    
    except Exception as e:
        print(f"💥 Quick memory test failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)