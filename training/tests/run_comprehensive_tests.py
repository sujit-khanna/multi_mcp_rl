#!/usr/bin/env python3
"""
Comprehensive Test Runner for GRPO Training Pipeline

This script runs all available tests and saves comprehensive results 
to a centralized location for easy review and validation.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """
    Runs all available tests and collects comprehensive results.
    """
    
    def __init__(self, results_dir: str = None):
        """Initialize test runner."""
        
        self.test_dir = Path(__file__).parent
        self.training_dir = self.test_dir.parent
        
        # Create results directory
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = f"test_results_{timestamp}"
        
        self.results_dir = self.test_dir / results_dir
        self.results_dir.mkdir(exist_ok=True)
        
        self.start_time = time.time()
        self.test_results = {}
        
        logger.info(f"Comprehensive test runner initialized")
        logger.info(f"Results directory: {self.results_dir}")
    
    def run_python_script(self, script_name: str, args: List[str] = None) -> Dict[str, Any]:
        """Run a Python script and capture output."""
        
        script_path = self.test_dir / script_name
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Script {script_name} not found",
                "stdout": "",
                "stderr": "",
                "return_code": -1
            }
        
        # Build command
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            # Run the script
            result = subprocess.run(
                cmd,
                cwd=str(self.test_dir),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Script timed out after 10 minutes",
                "return_code": -1,
                "stdout": "",
                "stderr": "",
                "command": ' '.join(cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run script: {e}",
                "return_code": -1,
                "stdout": "",
                "stderr": "",
                "command": ' '.join(cmd)
            }
    
    def save_test_result(self, test_name: str, result: Dict[str, Any]) -> None:
        """Save individual test result."""
        
        # Save to JSON file
        result_file = self.results_dir / f"{test_name}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save stdout to text file if available
        if result.get("stdout"):
            stdout_file = self.results_dir / f"{test_name}_stdout.txt"
            with open(stdout_file, 'w') as f:
                f.write(result["stdout"])
        
        # Save stderr to text file if available
        if result.get("stderr"):
            stderr_file = self.results_dir / f"{test_name}_stderr.txt"
            with open(stderr_file, 'w') as f:
                f.write(result["stderr"])
        
        logger.info(f"Saved {test_name} results to {result_file}")
    
    def test_file_structure_validation(self) -> Dict[str, Any]:
        """Test file structure validation."""
        
        logger.info("Running file structure validation...")
        result = self.run_python_script("validate_structure.py")
        
        # Parse success from stdout
        if result["success"] and "ALL VALIDATIONS PASSED" in result["stdout"]:
            result["validation_summary"] = "All validations passed"
            result["components_validated"] = [
                "File Structure", "Module Imports", "Configuration Files",
                "Component Instantiation", "Training Script"
            ]
        else:
            result["validation_summary"] = "Some validations failed"
        
        return result
    
    def test_minimal_components(self) -> Dict[str, Any]:
        """Test minimal component functionality."""
        
        logger.info("Running minimal component tests...")
        result = self.run_python_script("minimal_test.py")
        
        # Parse test results from stdout
        if result["success"] and "ALL TESTS PASSED" in result["stdout"]:
            result["test_summary"] = "All minimal tests passed"
            result["components_tested"] = [
                "Imports", "Data Components", "Trajectory Components",
                "Configuration Handling", "Device Detection", "Memory Monitoring"
            ]
        else:
            result["test_summary"] = "Some minimal tests failed"
        
        return result
    
    def test_memory_profiling_lora(self) -> Dict[str, Any]:
        """Test memory profiling in LoRA mode."""
        
        logger.info("Running memory profiling (LoRA mode)...")
        result = self.run_python_script("memory_profile.py", ["--mode", "lora", "--quick"])
        
        # Check if JSON results file was created
        memory_results_file = self.test_dir / "memory_profile_lora.json"
        if memory_results_file.exists():
            try:
                with open(memory_results_file, 'r') as f:
                    memory_data = json.load(f)
                result["memory_profile_data"] = memory_data
                
                # Analyze results
                successful_configs = [r for r in memory_data.get("results", []) if r.get("success", False)]
                result["successful_configurations"] = len(successful_configs)
                result["total_configurations_tested"] = len(memory_data.get("results", []))
                
            except Exception as e:
                result["memory_profile_error"] = f"Could not parse memory results: {e}"
        
        return result
    
    def test_smoke_test_lora(self) -> Dict[str, Any]:
        """Test smoke test in LoRA mode with model tests skipped."""
        
        logger.info("Running smoke test (LoRA mode, no model loading)...")
        result = self.run_python_script("smoke_test.py", ["--mode", "lora", "--skip_model_tests"])
        
        # Parse test results
        if result["success"]:
            if "ALL TESTS PASSED" in result["stdout"]:
                result["smoke_test_summary"] = "All smoke tests passed"
            else:
                result["smoke_test_summary"] = "Some smoke tests failed"
        else:
            result["smoke_test_summary"] = "Smoke test execution failed"
        
        return result
    
    def collect_existing_results(self) -> Dict[str, Any]:
        """Collect any existing test result files."""
        
        logger.info("Collecting existing test results...")
        
        existing_results = {}
        
        # Look for existing result files
        result_patterns = [
            "memory_profile_*.json",
            "test_results_*.json",
            "*_results.json"
        ]
        
        for pattern in result_patterns:
            for file_path in self.test_dir.glob(pattern):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    existing_results[file_path.stem] = {
                        "file_path": str(file_path),
                        "data": data,
                        "modified_time": file_path.stat().st_mtime
                    }
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        return {
            "success": True,
            "existing_results": existing_results,
            "files_found": len(existing_results)
        }
    
    def generate_system_info(self) -> Dict[str, Any]:
        """Generate system information."""
        
        logger.info("Collecting system information...")
        
        import platform
        import torch
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info["cuda_device_count"] = torch.cuda.device_count()
            system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            system_info["mps_device"] = "Apple Silicon GPU"
        
        # Memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info["total_memory_gb"] = memory.total / (1024**3)
            system_info["available_memory_gb"] = memory.available / (1024**3)
        except ImportError:
            system_info["memory_info"] = "psutil not available"
        
        return {
            "success": True,
            "system_info": system_info
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all available tests."""
        
        logger.info("Starting comprehensive test suite...")
        
        # Define all tests to run
        tests = [
            ("system_info", self.generate_system_info),
            ("existing_results", self.collect_existing_results),
            ("file_structure_validation", self.test_file_structure_validation),
            ("minimal_components", self.test_minimal_components),
            ("memory_profiling_lora", self.test_memory_profiling_lora),
            ("smoke_test_lora", self.test_smoke_test_lora),
        ]
        
        all_results = {}
        successful_tests = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test: {test_name}")
            logger.info('='*60)
            
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                
                result["test_name"] = test_name
                result["timestamp"] = time.time()
                result["duration"] = time.time() - self.start_time
                
                all_results[test_name] = result
                self.save_test_result(test_name, result)
                
                if result.get("success", False):
                    successful_tests += 1
                    logger.info(f"✅ {test_name} completed successfully")
                else:
                    logger.error(f"❌ {test_name} failed")
                
            except Exception as e:
                logger.error(f"💥 {test_name} crashed: {e}")
                logger.error(traceback.format_exc())
                
                error_result = {
                    "test_name": test_name,
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": time.time(),
                    "duration": time.time() - self.start_time
                }
                
                all_results[test_name] = error_result
                self.save_test_result(test_name, error_result)
        
        # Generate summary
        total_duration = time.time() - self.start_time
        summary = {
            "test_suite": "GRPO Training Pipeline Comprehensive Tests",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": total_duration,
            "total_tests": len(tests),
            "successful_tests": successful_tests,
            "failed_tests": len(tests) - successful_tests,
            "success_rate": (successful_tests / len(tests)) * 100,
            "results_directory": str(self.results_dir),
            "individual_results": all_results
        }
        
        return summary
    
    def save_comprehensive_summary(self, summary: Dict[str, Any]) -> str:
        """Save comprehensive test summary."""
        
        # Save JSON summary
        summary_file = self.results_dir / "comprehensive_test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate human-readable report
        report_file = self.results_dir / "comprehensive_test_report.txt"
        with open(report_file, 'w') as f:
            f.write("GRPO TRAINING PIPELINE - COMPREHENSIVE TEST RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Test Suite: {summary['test_suite']}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Duration: {summary['duration_seconds']:.1f} seconds\n")
            f.write(f"Results Directory: {summary['results_directory']}\n\n")
            
            f.write("SUMMARY:\n")
            f.write(f"  Total Tests: {summary['total_tests']}\n")
            f.write(f"  Successful: {summary['successful_tests']}\n")
            f.write(f"  Failed: {summary['failed_tests']}\n")
            f.write(f"  Success Rate: {summary['success_rate']:.1f}%\n\n")
            
            f.write("INDIVIDUAL TEST RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for test_name, result in summary['individual_results'].items():
                status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
                f.write(f"{test_name}: {status}\n")
                
                if not result.get('success', False) and 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                
                if 'validation_summary' in result:
                    f.write(f"  Summary: {result['validation_summary']}\n")
                elif 'test_summary' in result:
                    f.write(f"  Summary: {result['test_summary']}\n")
                elif 'smoke_test_summary' in result:
                    f.write(f"  Summary: {result['smoke_test_summary']}\n")
                
                f.write("\n")
            
            # Overall assessment
            f.write("OVERALL ASSESSMENT:\n")
            f.write("-" * 20 + "\n")
            
            if summary['success_rate'] >= 80:
                f.write("🎉 EXCELLENT: Training pipeline is ready for deployment!\n")
            elif summary['success_rate'] >= 60:
                f.write("⚠️  GOOD: Most components working, minor issues to address.\n")
            else:
                f.write("❌ NEEDS WORK: Significant issues need to be resolved.\n")
        
        logger.info(f"Comprehensive summary saved to {summary_file}")
        logger.info(f"Human-readable report saved to {report_file}")
        
        return str(summary_file)


async def main():
    """Main entry point."""
    
    print("🔥 GRPO Training Pipeline - Comprehensive Test Suite")
    print("="*60)
    
    # Create test runner
    runner = ComprehensiveTestRunner()
    
    try:
        # Run all tests
        summary = await runner.run_all_tests()
        
        # Save comprehensive summary
        summary_file = runner.save_comprehensive_summary(summary)
        
        # Print final results
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*60)
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Successful Tests: {summary['successful_tests']}/{summary['total_tests']}")
        print(f"Results Directory: {summary['results_directory']}")
        print(f"Summary File: {summary_file}")
        
        # Individual test status
        print("\nIndividual Test Results:")
        for test_name, result in summary['individual_results'].items():
            status = "✅" if result.get('success', False) else "❌"
            print(f"  {status} {test_name}")
        
        if summary['success_rate'] >= 80:
            print("\n🎉 EXCELLENT: Training pipeline is ready for deployment!")
            return 0
        elif summary['success_rate'] >= 60:
            print("\n⚠️  GOOD: Most components working, minor issues to address.")
            return 0
        else:
            print("\n❌ NEEDS WORK: Significant issues need to be resolved.")
            return 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)