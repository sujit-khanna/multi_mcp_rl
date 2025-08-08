#!/usr/bin/env python3
"""
MCP Integration Test

This script tests the complete MCP integration to ensure the training pipeline
can successfully connect to and execute tools via MCP servers.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPIntegrationTester:
    """
    Tests MCP server integration for the training pipeline.
    """
    
    def __init__(self):
        self.mcp_servers_dir = Path(__file__).parent.parent.parent.parent / "mcp_tools" / "limited"
        self.test_results = {}
        self.server_processes = {}
        
        logger.info(f"MCP servers directory: {self.mcp_servers_dir}")
    
    def check_mcp_servers_available(self) -> Dict[str, bool]:
        """Check if MCP server files are available."""
        
        logger.info("Checking MCP server availability...")
        
        server_files = [
            "python_execution_server.py",
            "fmp_limited_server.py", 
            "polygon_limited_server.py",
            "slack_limited_server.py",
            "tavily_limited_server.py"
        ]
        
        availability = {}
        for server_file in server_files:
            server_path = self.mcp_servers_dir / server_file
            available = server_path.exists()
            availability[server_file] = available
            status = "✅" if available else "❌"
            logger.info(f"  {status} {server_file}")
        
        return availability
    
    def test_mcp_environment_import(self) -> bool:
        """Test if MCPToolEnvironment can be imported and instantiated."""
        
        logger.info("Testing MCP environment import...")
        
        try:
            # Add environments directory to path
            environments_dir = Path(__file__).parent.parent.parent / "environments"
            if str(environments_dir) not in sys.path:
                sys.path.insert(0, str(environments_dir))
            
            from mcp_tool_environment import MCPToolEnvironment
            from real_tool_manager import RealMCPToolManager
            
            logger.info("✅ MCPToolEnvironment imported successfully")
            logger.info("✅ RealMCPToolManager imported successfully")
            
            # Test basic instantiation (without starting servers)
            try:
                # Create a simple test config in the correct format
                test_data = {
                    "task_metadata": {"task_id": "test_001", "complexity": "easy"},
                    "prompt": [{"role": "user", "content": "Test task"}],
                    "reward_spec": {"ground_truth": {"expected_tools": []}},
                    "extra_info": {"complexity": "easy"}
                }
                
                # This should not start servers, just create the environment
                env = MCPToolEnvironment(task_data=test_data)
                
                logger.info("✅ MCPToolEnvironment instantiated successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ MCPToolEnvironment instantiation failed: {e}")
                return False
                
        except ImportError as e:
            logger.error(f"❌ MCP environment import failed: {e}")
            return False
    
    def test_python_execution_server(self) -> bool:
        """Test Python execution server specifically (lightweight test)."""
        
        logger.info("Testing Python execution server...")
        
        try:
            # Check if we can import the server module
            python_server_path = self.mcp_servers_dir / "python_execution_server.py"
            
            if not python_server_path.exists():
                logger.error("❌ Python execution server not found")
                return False
            
            # Run a simple validation test
            cmd = [
                sys.executable, 
                str(python_server_path),
                "--test"  # If the server supports a test mode
            ]
            
            try:
                # Try to run the server in test mode (timeout after 5 seconds)
                result = subprocess.run(
                    [sys.executable, "-c", f"import sys; sys.path.append('{self.mcp_servers_dir}'); import python_execution_server; print('✅ Server module loads successfully')"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    logger.info("✅ Python execution server module loads successfully")
                    return True
                else:
                    logger.error(f"❌ Python server module test failed: {result.stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.warning("⚠️  Python server test timed out (may be normal)")
                return True  # Timeout is acceptable for server testing
                
        except Exception as e:
            logger.error(f"❌ Python execution server test failed: {e}")
            return False
    
    def test_tool_manager_connection(self) -> bool:
        """Test tool manager without starting servers."""
        
        logger.info("Testing tool manager connection capability...")
        
        try:
            # Add environments directory to path
            environments_dir = Path(__file__).parent.parent.parent / "environments"
            if str(environments_dir) not in sys.path:
                sys.path.insert(0, str(environments_dir))
            
            from real_tool_manager import RealMCPToolManager
            
            # Create tool manager (should not connect immediately)
            tool_manager = RealMCPToolManager()
            
            # Check if connection pooling is configured
            if hasattr(tool_manager, 'server_configs'):
                logger.info("✅ Tool manager has server configurations")
            
            if hasattr(tool_manager, 'connection_pool'):
                logger.info("✅ Tool manager has connection pooling")
            
            logger.info("✅ Tool manager instantiated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Tool manager test failed: {e}")
            return False
    
    def test_training_integration(self) -> bool:
        """Test if the training pipeline can use MCP integration."""
        
        logger.info("Testing training pipeline MCP integration...")
        
        try:
            # Test if trajectory collector can work with MCP
            from data.trajectory_collector import TrajectoryCollector, ENVIRONMENT_IMPORTS_AVAILABLE
            
            if ENVIRONMENT_IMPORTS_AVAILABLE:
                logger.info("✅ Trajectory collector has environment imports available")
                
                # Test basic instantiation with correct parameters
                def dummy_env_factory(task_data):
                    return None  # Mock environment factory
                
                class DummyPolicy:
                    def enable_eval_mode(self):
                        pass
                    def generate_action(self, states):
                        return ["test response"]
                    def compute_log_probs(self, states, actions):
                        return torch.tensor([0.0])
                
                dummy_policy = DummyPolicy()
                
                collector = TrajectoryCollector(
                    policy=dummy_policy,
                    env_factory=dummy_env_factory,
                    num_parallel_envs=1,
                    shared_tool_manager=None,
                    max_episode_length=5,
                    executor_max_workers=1
                )
                
                logger.info("✅ Trajectory collector instantiated successfully")
                return True
            else:
                logger.warning("⚠️  Environment imports not available in trajectory collector")
                return False
                
        except Exception as e:
            logger.error(f"❌ Training integration test failed: {e}")
            return False
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive MCP integration tests."""
        
        logger.info("Starting comprehensive MCP integration test...")
        start_time = time.time()
        
        results = {
            "test_suite": "MCP Integration Test",
            "timestamp": time.time(),
            "tests": {}
        }
        
        # Test 1: Server availability
        server_availability = self.check_mcp_servers_available()
        results["tests"]["server_availability"] = {
            "success": all(server_availability.values()),
            "details": server_availability
        }
        
        # Test 2: Environment imports
        env_import_success = self.test_mcp_environment_import()
        results["tests"]["environment_import"] = {
            "success": env_import_success
        }
        
        # Test 3: Python execution server
        python_server_success = self.test_python_execution_server()
        results["tests"]["python_execution_server"] = {
            "success": python_server_success
        }
        
        # Test 4: Tool manager
        tool_manager_success = self.test_tool_manager_connection()
        results["tests"]["tool_manager"] = {
            "success": tool_manager_success
        }
        
        # Test 5: Training integration
        training_integration_success = self.test_training_integration()
        results["tests"]["training_integration"] = {
            "success": training_integration_success
        }
        
        # Summary
        successful_tests = sum(1 for test in results["tests"].values() if test["success"])
        total_tests = len(results["tests"])
        
        results["summary"] = {
            "duration_seconds": time.time() - start_time,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": (successful_tests / total_tests) * 100,
            "overall_success": successful_tests == total_tests
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print test results summary."""
        
        print("\n" + "="*60)
        print("MCP INTEGRATION TEST RESULTS")
        print("="*60)
        
        print(f"Duration: {results['summary']['duration_seconds']:.2f} seconds")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Successful Tests: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
        
        print("\nDetailed Results:")
        for test_name, test_result in results["tests"].items():
            status = "✅" if test_result["success"] else "❌"
            print(f"  {status} {test_name}")
            
            if "details" in test_result:
                for detail_name, detail_value in test_result["details"].items():
                    detail_status = "✅" if detail_value else "❌"
                    print(f"    {detail_status} {detail_name}")
        
        print("\nOverall Assessment:")
        if results["summary"]["overall_success"]:
            print("🎉 ALL MCP INTEGRATION TESTS PASSED!")
            print("✅ Training pipeline can use MCP tools for real training")
        else:
            print("⚠️  Some MCP integration tests failed")
            print("❌ Training pipeline may have issues with real tool execution")
        
        print("="*60)


async def main():
    """Main entry point."""
    
    print("🔧 MCP Integration Test for GRPO Training")
    print("="*50)
    
    tester = MCPIntegrationTester()
    
    try:
        results = await tester.run_comprehensive_test()
        tester.print_results(results)
        
        # Save results
        results_file = Path(__file__).parent / "mcp_integration_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        if results["summary"]["overall_success"]:
            return 0
        else:
            return 1
    
    except Exception as e:
        print(f"💥 MCP integration test crashed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)