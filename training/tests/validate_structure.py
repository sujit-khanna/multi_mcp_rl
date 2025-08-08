#!/usr/bin/env python3
"""
Validate GRPO Training Structure

This script validates that the complete GRPO training structure is properly
implemented and all components can be imported and initialized without errors.
"""

import sys
import json
import tempfile
import traceback
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

def validate_file_structure():
    """Validate that all expected files exist."""
    print("🔍 Validating file structure...")
    
    base_path = Path(__file__).parent.parent
    
    expected_files = [
        # Core components
        "core/__init__.py",
        "core/qwen_policy.py", 
        "core/grpo_trainer.py",
        
        # Data components
        "data/__init__.py",
        "data/data_loader.py",
        "data/trajectory_collector.py",
        
        # Configuration files
        "configs/model_config.yaml",
        "configs/training_config.yaml", 
        "configs/grpo_config.yaml",
        "configs/accelerate_config.yaml",
        "configs/deepspeed_config.json",
        
        # Scripts
        "scripts/__init__.py",
        "scripts/train_grpo.py",
        
        # Tests
        "tests/smoke_test.py",
        "tests/memory_profile.py",
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in expected_files:
        full_path = base_path / file_path
        if full_path.exists():
            existing_files.append(file_path)
            print(f"  ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}")
    
    print(f"\nFile structure validation: {len(existing_files)}/{len(expected_files)} files found")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All expected files present")
        return True

def validate_imports():
    """Validate that all modules can be imported."""
    print("\n🔍 Validating imports...")
    
    imports_to_test = [
        ("core.qwen_policy", ["QwenPolicy", "create_policy_from_configs"]),
        ("core.grpo_trainer", ["GRPOTrainer", "Trajectory", "create_grpo_trainer"]),
        ("data.data_loader", ["StreamingDataset", "CurriculumSampler", "TaskBatcher", "TaskDataLoader"]),
        ("data.trajectory_collector", ["TrajectoryCollector", "EpisodeResult"]),
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module_name, expected_classes in imports_to_test:
        try:
            module = __import__(module_name, fromlist=expected_classes)
            
            missing_classes = []
            for class_name in expected_classes:
                if not hasattr(module, class_name):
                    missing_classes.append(class_name)
            
            if missing_classes:
                print(f"  ⚠️  {module_name}: missing {missing_classes}")
                failed_imports.append((module_name, missing_classes))
            else:
                print(f"  ✅ {module_name}: all classes available")
                successful_imports.append(module_name)
                
        except Exception as e:
            print(f"  ❌ {module_name}: import failed - {e}")
            failed_imports.append((module_name, str(e)))
    
    print(f"\nImport validation: {len(successful_imports)}/{len(imports_to_test)} modules imported successfully")
    
    if failed_imports:
        print(f"❌ Failed imports: {failed_imports}")
        return False
    else:
        print("✅ All modules imported successfully")
        return True

def validate_configurations():
    """Validate that configuration files are valid."""
    print("\n🔍 Validating configuration files...")
    
    import yaml
    
    base_path = Path(__file__).parent.parent
    config_files = [
        ("configs/model_config.yaml", "yaml"),
        ("configs/training_config.yaml", "yaml"),
        ("configs/grpo_config.yaml", "yaml"),
        ("configs/accelerate_config.yaml", "yaml"),
        ("configs/deepspeed_config.json", "json"),
    ]
    
    valid_configs = []
    invalid_configs = []
    
    for config_file, file_type in config_files:
        try:
            config_path = base_path / config_file
            
            if not config_path.exists():
                print(f"  ❌ {config_file}: file not found")
                invalid_configs.append(config_file)
                continue
            
            with open(config_path, 'r') as f:
                if file_type == "yaml":
                    config_data = yaml.safe_load(f)
                else:  # json
                    config_data = json.load(f)
            
            if config_data and isinstance(config_data, dict):
                print(f"  ✅ {config_file}: valid {file_type}")
                valid_configs.append(config_file)
            else:
                print(f"  ❌ {config_file}: invalid format")
                invalid_configs.append(config_file)
                
        except Exception as e:
            print(f"  ❌ {config_file}: parsing failed - {e}")
            invalid_configs.append(config_file)
    
    print(f"\nConfiguration validation: {len(valid_configs)}/{len(config_files)} files valid")
    
    if invalid_configs:
        print(f"❌ Invalid configs: {invalid_configs}")
        return False
    else:
        print("✅ All configuration files valid")
        return True

def validate_component_instantiation():
    """Validate that components can be instantiated with mock data."""
    print("\n🔍 Validating component instantiation...")
    
    try:
        # Test data components
        from data.data_loader import CurriculumSampler, TaskBatcher, LRUCache
        
        # LRU Cache
        cache = LRUCache(capacity=10)
        cache.put("test", "value")
        assert cache.get("test") == "value"
        print("  ✅ LRUCache instantiated and working")
        
        # Curriculum Sampler  
        sampler = CurriculumSampler()
        dist = sampler.get_current_distribution(0, 10)
        assert isinstance(dist, dict) and abs(sum(dist.values()) - 1.0) < 1e-6
        print("  ✅ CurriculumSampler instantiated and working")
        
        # Task Batcher
        batcher = TaskBatcher()
        sample_tasks = [{"task_metadata": {"complexity": "easy"}}]
        batch = batcher.create_batch(sample_tasks)
        assert len(batch) > 0
        print("  ✅ TaskBatcher instantiated and working")
        
        # Test trajectory components
        from core.grpo_trainer import Trajectory
        from data.trajectory_collector import EpisodeResult
        
        # Trajectory
        trajectory = Trajectory(
            task_id="test",
            states=[[{"role": "user", "content": "test"}]],
            actions=["response"],
            rewards=[1.0],
            dones=[True]
        )
        assert trajectory.total_reward == 1.0
        print("  ✅ Trajectory instantiated and working")
        
        # EpisodeResult
        episode = EpisodeResult(
            task_id="test",
            trajectory=[{
                "turn": 1,
                "action": "test",
                "observation": "test",
                "reward": 1.0,
                "done": True,
                "metadata": {}
            }],
            total_reward=1.0,
            turns=1,
            success=True
        )
        assert episode.is_valid()
        print("  ✅ EpisodeResult instantiated and working")
        
        print("\n✅ All components can be instantiated")
        return True
        
    except Exception as e:
        print(f"\n❌ Component instantiation failed: {e}")
        print(traceback.format_exc())
        return False

def validate_training_script():
    """Validate that the main training script can be parsed."""
    print("\n🔍 Validating training script...")
    
    try:
        script_path = Path(__file__).parent.parent / "scripts" / "train_grpo.py"
        
        if not script_path.exists():
            print("  ❌ train_grpo.py not found")
            return False
        
        # Try to parse the script (without executing)
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Basic validation
        required_components = [
            "TrainingSession",
            "parse_arguments", 
            "main",
            "argparse",
            "asyncio"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in script_content:
                missing_components.append(component)
        
        if missing_components:
            print(f"  ❌ Missing components in train_grpo.py: {missing_components}")
            return False
        
        # Try to compile the script
        compile(script_content, str(script_path), 'exec')
        
        print("  ✅ train_grpo.py is syntactically valid")
        return True
        
    except Exception as e:
        print(f"  ❌ Training script validation failed: {e}")
        return False

def run_validation():
    """Run complete validation suite."""
    print("🔥 GRPO Training Structure Validation")
    print("="*50)
    
    validations = [
        ("File Structure", validate_file_structure),
        ("Module Imports", validate_imports),
        ("Configuration Files", validate_configurations), 
        ("Component Instantiation", validate_component_instantiation),
        ("Training Script", validate_training_script),
    ]
    
    results = {}
    passed = 0
    total = len(validations)
    
    for validation_name, validation_func in validations:
        try:
            success = validation_func()
            results[validation_name] = success
            if success:
                passed += 1
        except Exception as e:
            print(f"\n❌ {validation_name} validation crashed: {e}")
            results[validation_name] = False
    
    # Print final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for validation_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {validation_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ GRPO training structure is complete and ready for use")
        return True
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed")
        print("❌ Some components need attention before training")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)