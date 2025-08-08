#!/usr/bin/env python3
"""
Extended Dataset Generator for SkyRL-Compatible Curriculum Learning
================================================================

Processes extended datasets with complexity levels and task metadata for 
progressive RL training. Supports curriculum learning through adaptive 
difficulty progression and comprehensive performance tracking.

Compatible with SkyRL format:
https://skyrl.readthedocs.io/en/latest/datasets/dataset-preparation.html

Author: MCP Tools Team
Date: 2025-01-25
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Keep system awake during long operations
try:
    from wakepy import keep
    WAKEPY_AVAILABLE = True
except ImportError:
    print("⚠️  wakepy not available. Install with: pip install wakepy")
    WAKEPY_AVAILABLE = False

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# Import the enhanced mini agent using importlib to handle filename with spaces
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mini_agent_module", 
    current_dir.parent / "mini_agent_no_fallback copy.py"
)
mini_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mini_agent_module)

# Import required classes and constants
MiniAgent = mini_agent_module.MiniAgent
COMPLEXITY_CONFIG = mini_agent_module.COMPLEXITY_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
generator_logger = logger.getChild("generator")
curriculum_logger = logger.getChild("curriculum")
analysis_logger = logger.getChild("analysis")


class ExtendedDatasetGenerator:
    """Advanced dataset generator with curriculum learning support and periodic saving"""
    
    def __init__(self, input_dataset_path: str):
        """Initialize the generator with input dataset"""
        self.input_dataset_path = input_dataset_path
        self.agent = None
        self.dataset_tasks = []
        self.complexity_distribution = {"easy": 0, "medium": 0, "hard": 0}
        self.category_distribution = {}
        
    async def initialize(self):
        """Initialize the MiniAgent and load input dataset"""
        generator_logger.info("Initializing Extended Dataset Generator...")
        
        # Initialize agent
        self.agent = MiniAgent()
        await self.agent.tool_manager.initialize_tools()
        
        # Load input dataset
        await self._load_input_dataset()
        
        generator_logger.info(f"Generator initialized with {len(self.dataset_tasks)} tasks")
        self._log_dataset_statistics()
    
    async def _load_input_dataset(self):
        """Load and validate the input dataset"""
        try:
            with open(self.input_dataset_path, 'r') as f:
                self.dataset_tasks = json.load(f)
            
            # Validate dataset structure
            required_fields = ["task_id", "category", "complexity", "question"]
            for i, task in enumerate(self.dataset_tasks):
                missing_fields = [field for field in required_fields if field not in task]
                if missing_fields:
                    raise ValueError(f"Task {i+1} missing required fields: {missing_fields}")
            
            # Build distribution statistics
            for task in self.dataset_tasks:
                complexity = task["complexity"]
                category = task["category"]
                
                self.complexity_distribution[complexity] = self.complexity_distribution.get(complexity, 0) + 1
                self.category_distribution[category] = self.category_distribution.get(category, 0) + 1
            
            generator_logger.info(f"Loaded {len(self.dataset_tasks)} tasks from {self.input_dataset_path}")
            
        except Exception as e:
            generator_logger.error(f"Failed to load input dataset: {e}")
            raise
    
    def _log_dataset_statistics(self):
        """Log comprehensive dataset statistics"""
        total_tasks = len(self.dataset_tasks)
        
        generator_logger.info("📊 Input Dataset Statistics:")
        generator_logger.info(f"  • Total tasks: {total_tasks}")
        
        generator_logger.info("  • Complexity distribution:")
        for complexity, count in self.complexity_distribution.items():
            percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
            generator_logger.info(f"    - {complexity}: {count} ({percentage:.1f}%)")
        
        generator_logger.info("  • Category distribution:")
        for category, count in sorted(self.category_distribution.items()):
            percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
            generator_logger.info(f"    - {category}: {count} ({percentage:.1f}%)")
    
    def filter_tasks(self, 
                    categories: Optional[List[str]] = None,
                    complexity_levels: Optional[List[str]] = None,
                    task_ids: Optional[List[str]] = None,
                    max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """Filter tasks based on various criteria"""
        filtered_tasks = self.dataset_tasks.copy()
        
        # Filter by categories
        if categories:
            filtered_tasks = [task for task in filtered_tasks if task["category"] in categories]
        
        # Filter by complexity levels
        if complexity_levels:
            filtered_tasks = [task for task in filtered_tasks if task["complexity"] in complexity_levels]
        
        # Filter by specific task IDs
        if task_ids:
            filtered_tasks = [task for task in filtered_tasks if task["task_id"] in task_ids]
        
        # Limit number of tasks
        if max_tasks:
            filtered_tasks = filtered_tasks[:max_tasks]
        
        generator_logger.info(f"Filtered to {len(filtered_tasks)} tasks")
        return filtered_tasks

    async def _save_intermediate_results(self, dataset: List[Dict[str, Any]], output_path: str, 
                                       detailed_log: List[Dict[str, Any]], batch_num: int, 
                                       total_batches: int):
        """Save intermediate results every 5 questions"""
        try:
            # Save main dataset
            temp_output = output_path.replace('.json', f'_batch_{batch_num}_of_{total_batches}.json')
            with open(temp_output, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            # Save detailed log
            temp_log = output_path.replace('.json', f'_batch_{batch_num}_of_{total_batches}_detailed_log.json')
            log_data = {
                "generation_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "batch_number": batch_num,
                    "total_batches": total_batches,
                    "entries_in_batch": len(dataset),
                    "partial_generation": True,
                    "skyrl_format_version": "1.0_extended"
                },
                "queries": detailed_log
            }
            with open(temp_log, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            generator_logger.info(f"💾 Saved intermediate results: batch {batch_num}/{total_batches}")
            generator_logger.info(f"  • Dataset: {temp_output}")
            generator_logger.info(f"  • Log: {temp_log}")
            
        except Exception as e:
            generator_logger.error(f"Failed to save intermediate results: {e}")

    async def generate_curriculum_dataset_with_periodic_saving(self,
                                                             output_path: str,
                                                             categories: Optional[List[str]] = None,
                                                             complexity_levels: Optional[List[str]] = None,
                                                             curriculum_order: bool = True,
                                                             max_tasks: Optional[int] = None,
                                                             batch_size: int = 5) -> Dict[str, Any]:
        """Generate SkyRL-compatible dataset with periodic saving every batch_size questions"""
        
        curriculum_logger.info("🎓 Starting curriculum dataset generation with periodic saving...")
        
        # Filter tasks based on criteria
        tasks_to_process = self.filter_tasks(
            categories=categories,
            complexity_levels=complexity_levels,
            max_tasks=max_tasks
        )
        
        # Sort tasks for curriculum learning if requested
        if curriculum_order:
            tasks_to_process = self._sort_for_curriculum_learning(tasks_to_process)
        
        # Prepare batches
        total_tasks = len(tasks_to_process)
        total_batches = (total_tasks + batch_size - 1) // batch_size  # Ceiling division
        
        curriculum_logger.info(f"Processing {total_tasks} tasks in {total_batches} batches of {batch_size}")
        
        # Initialize accumulation variables
        all_dataset_entries = []
        all_detailed_logs = []
        
        # Use wakepy to keep system awake if available
        async def _generate_with_wakepy():
            for batch_num in range(1, total_batches + 1):
                start_idx = (batch_num - 1) * batch_size
                end_idx = min(start_idx + batch_size, total_tasks)
                batch_tasks = tasks_to_process[start_idx:end_idx]
                
                curriculum_logger.info(f"🔄 Processing batch {batch_num}/{total_batches} (tasks {start_idx + 1}-{end_idx})")
                
                # Prepare batch data
                queries = [task["question"] for task in batch_tasks]
                complexity_levels_list = [task["complexity"] for task in batch_tasks]
                task_metadata_list = [
                    {
                        "task_id": task["task_id"],
                        "category": task["category"],
                        "complexity": task["complexity"],
                        "question": task["question"],
                        "prerequisite_categories": self._get_prerequisite_categories(task["category"]),
                        "skill_level": COMPLEXITY_CONFIG[task["complexity"]]["curriculum_score"]
                    }
                    for task in batch_tasks
                ]
                
                # Generate batch using agent
                batch_output = f"temp_batch_{batch_num}.json"
                batch_dataset = await self.agent.generate_dataset(
                    queries=queries,
                    output_file=batch_output,
                    complexity_levels=complexity_levels_list,
                    task_metadata_list=task_metadata_list
                )
                
                # Read detailed log from the batch generation
                batch_log_file = batch_output.replace('.json', '_detailed_log.json')
                try:
                    with open(batch_log_file, 'r') as f:
                        batch_log_data = json.load(f)
                        batch_detailed_logs = batch_log_data.get("queries", [])
                except:
                    batch_detailed_logs = []
                
                # Accumulate results
                all_dataset_entries.extend(batch_dataset)
                all_detailed_logs.extend(batch_detailed_logs)
                
                # Save intermediate results
                await self._save_intermediate_results(
                    all_dataset_entries, output_path, all_detailed_logs, 
                    batch_num, total_batches
                )
                
                # Clean up temporary files
                try:
                    os.remove(batch_output)
                    if os.path.exists(batch_log_file):
                        os.remove(batch_log_file)
                except:
                    pass  # Ignore cleanup errors
                
                curriculum_logger.info(f"✅ Batch {batch_num}/{total_batches} complete ({len(batch_dataset)} entries)")
            
            return all_dataset_entries, all_detailed_logs
        
        # Execute with or without wakepy
        if WAKEPY_AVAILABLE:
            curriculum_logger.info("🔋 Using wakepy to keep system awake during generation")
            with keep.running():
                final_dataset, final_detailed_logs = await _generate_with_wakepy()
        else:
            curriculum_logger.warning("⚠️  wakepy not available - system may sleep during long operations")
            final_dataset, final_detailed_logs = await _generate_with_wakepy()
        
        # Save final results
        curriculum_logger.info("💾 Saving final consolidated results...")
        
        # Save final dataset
        with open(output_path, 'w') as f:
            json.dump(final_dataset, f, indent=2)
        
        # Save final detailed log
        final_log_path = output_path.replace('.json', '_detailed_log.json')
        final_log_data = {
            "generation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_entries_generated": len(final_dataset),
                "total_original_tasks": len(tasks_to_process),
                "generation_success_rate": (len(final_dataset) / len(tasks_to_process) * 100) if tasks_to_process else 0,
                "skyrl_format_version": "1.0_extended",
                "batch_size_used": batch_size,
                "total_batches_processed": total_batches,
                "periodic_saving_enabled": True
            },
            "queries": final_detailed_logs
        }
        
        with open(final_log_path, 'w') as f:
            json.dump(final_log_data, f, indent=2)
        
        # Generate comprehensive analysis
        analysis = self._analyze_generated_dataset(final_dataset, tasks_to_process)
        
        # Save analysis report
        analysis_path = output_path.replace('.json', '_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        curriculum_logger.info(f"✅ Generated {len(final_dataset)} entries with periodic saving")
        curriculum_logger.info(f"📄 Final dataset: {output_path}")
        curriculum_logger.info(f"📄 Final log: {final_log_path}")
        curriculum_logger.info(f"📄 Analysis: {analysis_path}")
        
        return {
            "dataset": final_dataset,
            "analysis": analysis,
            "tasks_processed": tasks_to_process,
            "detailed_logs": final_detailed_logs
        }

    async def generate_curriculum_dataset(self,
                                        output_path: str,
                                        categories: Optional[List[str]] = None,
                                        complexity_levels: Optional[List[str]] = None,
                                        curriculum_order: bool = True,
                                        max_tasks: Optional[int] = None) -> Dict[str, Any]:
        """Generate SkyRL-compatible dataset with curriculum learning progression"""
        
        # Use the new periodic saving method by default
        return await self.generate_curriculum_dataset_with_periodic_saving(
            output_path=output_path,
            categories=categories,
            complexity_levels=complexity_levels,
            curriculum_order=curriculum_order,
            max_tasks=max_tasks,
            batch_size=5  # Save every 5 questions
        )
    
    def _sort_for_curriculum_learning(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort tasks for optimal curriculum learning progression"""
        curriculum_logger.info("📚 Sorting tasks for curriculum learning progression...")
        
        # Define curriculum progression order
        complexity_order = {"easy": 1, "medium": 2, "hard": 3}
        category_progression = {
            # Simple market data queries
            "a": 1, "b": 2, "c": 3,
            # Analysis and reporting  
            "d": 4, "e": 5, "f": 6,
            # Advanced portfolio and risk management
            "g": 7, "h": 8, "i": 9, "j": 10
        }
        
        # Sort by curriculum progression: complexity first, then category
        sorted_tasks = sorted(tasks, key=lambda task: (
            complexity_order.get(task["complexity"], 99),
            category_progression.get(task["category"], 99),
            task["task_id"]  # Consistent ordering within same complexity/category
        ))
        
        curriculum_logger.info("✅ Tasks sorted for optimal learning progression")
        return sorted_tasks
    
    def _get_prerequisite_categories(self, category: str) -> List[str]:
        """Get prerequisite categories for curriculum learning"""
        prerequisites_map = {
            "a": [],                           # Basic queries - no prerequisites
            "b": ["a"],                        # Market data - needs basic queries
            "c": ["a", "b"],                   # Analysis - needs market data
            "d": ["a", "b"],                   # Dashboard - needs market data
            "e": ["a", "b", "c"],             # Advanced analysis - needs previous
            "f": ["a", "b", "c", "d"],        # Cross-asset - needs all previous
            "g": ["a", "b", "c"],             # Portfolio - needs analysis
            "h": ["a", "b", "c", "g"],        # Advanced portfolio - needs portfolio
            "i": ["a", "b", "c", "g"],        # Risk management - needs portfolio
            "j": ["a", "b", "c", "d", "e", "f", "g", "h", "i"]  # Expert - needs all
        }
        return prerequisites_map.get(category, [])
    
    def _analyze_generated_dataset(self, dataset: List[Dict[str, Any]], original_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive analysis of the generated dataset"""
        analysis_logger.info("📊 Analyzing generated dataset...")
        
        # Basic statistics
        total_entries = len(dataset)
        complexity_stats = {"easy": 0, "medium": 0, "hard": 0}
        category_stats = {}
        quality_stats = {"high": 0, "medium": 0, "low": 0}
        
        # Performance metrics
        curriculum_metrics = {
            "avg_reasoning_score_by_complexity": {},
            "avg_tool_calls_by_complexity": {},
            "success_rate_by_complexity": {},
            "skill_progression_indicators": {}
        }
        
        # Analyze each entry
        for entry in dataset:
            extra_info = entry.get("extra_info", {})
            
            # Complexity distribution
            complexity = extra_info.get("complexity", "unknown")
            complexity_stats[complexity] = complexity_stats.get(complexity, 0) + 1
            
            # Category distribution  
            task_metadata = extra_info.get("task_metadata", {})
            category = task_metadata.get("category", "unknown")
            category_stats[category] = category_stats.get(category, 0) + 1
            
            # Quality distribution
            quality_tier = extra_info.get("training_metadata", {}).get("quality_tier", "low")
            quality_stats[quality_tier] = quality_stats.get(quality_tier, 0) + 1
            
            # Curriculum metrics
            reasoning_quality = extra_info.get("reasoning_quality", {})
            execution_metrics = extra_info.get("execution_metrics", {})
            
            if complexity not in curriculum_metrics["avg_reasoning_score_by_complexity"]:
                curriculum_metrics["avg_reasoning_score_by_complexity"][complexity] = []
                curriculum_metrics["avg_tool_calls_by_complexity"][complexity] = []
                curriculum_metrics["success_rate_by_complexity"][complexity] = []
            
            curriculum_metrics["avg_reasoning_score_by_complexity"][complexity].append(
                reasoning_quality.get("best_score", 0)
            )
            curriculum_metrics["avg_tool_calls_by_complexity"][complexity].append(
                execution_metrics.get("num_tool_calls", 0)
            )
            curriculum_metrics["success_rate_by_complexity"][complexity].append(
                1 if extra_info.get("training_metadata", {}).get("curriculum_ready", False) else 0
            )
        
        # Calculate averages
        for complexity in curriculum_metrics["avg_reasoning_score_by_complexity"]:
            scores = curriculum_metrics["avg_reasoning_score_by_complexity"][complexity]
            tool_calls = curriculum_metrics["avg_tool_calls_by_complexity"][complexity]
            success_rates = curriculum_metrics["success_rate_by_complexity"][complexity]
            
            curriculum_metrics["avg_reasoning_score_by_complexity"][complexity] = sum(scores) / len(scores) if scores else 0
            curriculum_metrics["avg_tool_calls_by_complexity"][complexity] = sum(tool_calls) / len(tool_calls) if tool_calls else 0
            curriculum_metrics["success_rate_by_complexity"][complexity] = sum(success_rates) / len(success_rates) if success_rates else 0
        
        # Curriculum readiness assessment
        curriculum_readiness = self._assess_curriculum_readiness(dataset)
        
        analysis = {
            "generation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_entries_generated": total_entries,
                "total_original_tasks": len(original_tasks),
                "generation_success_rate": (total_entries / len(original_tasks) * 100) if original_tasks else 0,
                "skyrl_format_version": "1.0_extended"
            },
            "distribution_analysis": {
                "complexity_distribution": complexity_stats,
                "category_distribution": category_stats,
                "quality_distribution": quality_stats
            },
            "curriculum_metrics": curriculum_metrics,
            "curriculum_readiness": curriculum_readiness,
            "performance_summary": {
                "overall_success_rate": sum(curriculum_metrics["success_rate_by_complexity"].values()) / len(curriculum_metrics["success_rate_by_complexity"]) if curriculum_metrics["success_rate_by_complexity"] else 0,
                "avg_reasoning_score": sum(curriculum_metrics["avg_reasoning_score_by_complexity"].values()) / len(curriculum_metrics["avg_reasoning_score_by_complexity"]) if curriculum_metrics["avg_reasoning_score_by_complexity"] else 0,
                "avg_tool_usage": sum(curriculum_metrics["avg_tool_calls_by_complexity"].values()) / len(curriculum_metrics["avg_tool_calls_by_complexity"]) if curriculum_metrics["avg_tool_calls_by_complexity"] else 0
            },
            "recommendations": self._generate_training_recommendations(curriculum_metrics)
        }
        
        analysis_logger.info("✅ Dataset analysis complete")
        return analysis
    
    def _assess_curriculum_readiness(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess curriculum learning readiness of the dataset"""
        curriculum_logger.info("📋 Assessing curriculum learning readiness...")
        
        difficulty_progression = {"easy": [], "medium": [], "hard": []}
        skill_coverage = set()
        
        for entry in dataset:
            extra_info = entry.get("extra_info", {})
            complexity = extra_info.get("complexity", "unknown")
            curriculum_metadata = extra_info.get("curriculum_metadata", {})
            
            if complexity in difficulty_progression:
                difficulty_progression[complexity].append({
                    "task_id": extra_info.get("task_metadata", {}).get("task_id", "unknown"),
                    "success": extra_info.get("training_metadata", {}).get("curriculum_ready", False),
                    "reasoning_score": extra_info.get("reasoning_quality", {}).get("best_score", 0)
                })
            
            # Collect skill dependencies
            skills = curriculum_metadata.get("difficulty_progression", {}).get("skill_dependencies", [])
            skill_coverage.update(skills)
        
        # Calculate readiness metrics
        readiness_score = 0
        total_weight = 0
        
        for complexity, tasks in difficulty_progression.items():
            if tasks:
                complexity_success_rate = sum(1 for task in tasks if task["success"]) / len(tasks)
                weight = COMPLEXITY_CONFIG[complexity]["curriculum_score"]
                readiness_score += complexity_success_rate * weight
                total_weight += weight
        
        overall_readiness = (readiness_score / total_weight) if total_weight > 0 else 0
        
        return {
            "overall_readiness_score": overall_readiness,
            "difficulty_progression": difficulty_progression,
            "skill_coverage": sorted(list(skill_coverage)),
            "curriculum_recommendations": {
                "ready_for_training": overall_readiness >= 0.7,
                "suggested_training_order": ["easy", "medium", "hard"],
                "minimum_samples_per_level": {
                    "easy": max(10, len(difficulty_progression.get("easy", []))),
                    "medium": max(8, len(difficulty_progression.get("medium", []))),
                    "hard": max(5, len(difficulty_progression.get("hard", [])))
                }
            }
        }
    
    def _generate_training_recommendations(self, curriculum_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for RL training based on dataset analysis"""
        recommendations = []
        
        # Check success rates
        for complexity, success_rate in curriculum_metrics["success_rate_by_complexity"].items():
            if success_rate < 0.6:
                recommendations.append(
                    f"⚠️  Low success rate for {complexity} tasks ({success_rate:.1%}). "
                    f"Consider increasing timeout or adjusting complexity thresholds."
                )
            elif success_rate > 0.9:
                recommendations.append(
                    f"✅ Excellent success rate for {complexity} tasks ({success_rate:.1%}). "
                    f"Ready for RL training."
                )
        
        # Check reasoning quality
        for complexity, avg_score in curriculum_metrics["avg_reasoning_score_by_complexity"].items():
            threshold = COMPLEXITY_CONFIG[complexity]["min_reasoning_score"]
            if avg_score < threshold + 5:
                recommendations.append(
                    f"📝 Reasoning quality for {complexity} tasks could be improved "
                    f"(avg: {avg_score:.1f}, threshold: {threshold}). "
                    f"Consider fine-tuning reasoning prompts."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append("🎯 Dataset quality is excellent. Ready for curriculum RL training!")
        
        return recommendations
    
    async def generate_progressive_batches(self,
                                         output_dir: str,
                                         batch_size: int = 20,
                                         complexity_progression: List[str] = None) -> Dict[str, Any]:
        """Generate dataset in progressive batches for curriculum learning"""
        if complexity_progression is None:
            complexity_progression = ["easy", "medium", "hard"]
        
        os.makedirs(output_dir, exist_ok=True)
        batch_results = {}
        
        curriculum_logger.info(f"🔄 Generating progressive batches in {output_dir}")
        
        # Use wakepy if available for long operations
        async def _generate_batches():
            for complexity in complexity_progression:
                curriculum_logger.info(f"📚 Processing {complexity} complexity tasks...")
                
                # Filter tasks for this complexity level
                complexity_tasks = self.filter_tasks(
                    complexity_levels=[complexity],
                    max_tasks=batch_size
                )
                
                if not complexity_tasks:
                    curriculum_logger.warning(f"No {complexity} tasks found, skipping...")
                    continue
                
                # Generate batch with periodic saving
                batch_output_path = os.path.join(output_dir, f"curriculum_batch_{complexity}.json")
                batch_result = await self.generate_curriculum_dataset_with_periodic_saving(
                    output_path=batch_output_path,
                    complexity_levels=[complexity],
                    max_tasks=batch_size,
                    batch_size=5  # Save every 5 questions within each complexity batch
                )
                
                batch_results[complexity] = {
                    "output_path": batch_output_path,
                    "entries_generated": len(batch_result["dataset"]),
                    "analysis": batch_result["analysis"]
                }
                
                curriculum_logger.info(f"✅ {complexity} batch complete: {len(batch_result['dataset'])} entries")
            
            # Generate combined dataset
            combined_output_path = os.path.join(output_dir, "curriculum_combined.json")
            combined_result = await self.generate_curriculum_dataset_with_periodic_saving(
                output_path=combined_output_path,
                complexity_levels=complexity_progression,
                batch_size=5
            )
            
            batch_results["combined"] = {
                "output_path": combined_output_path,
                "entries_generated": len(combined_result["dataset"]),
                "analysis": combined_result["analysis"]
            }
            
            return batch_results
        
        # Execute with or without wakepy
        if WAKEPY_AVAILABLE:
            curriculum_logger.info("🔋 Using wakepy for progressive batch generation")
            with keep.running():
                results = await _generate_batches()
        else:
            results = await _generate_batches()
        
        curriculum_logger.info(f"🎉 Progressive batch generation complete!")
        return results


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Extended Dataset Generator for SkyRL Curriculum Learning")
    parser.add_argument("--input", required=True, help="Path to input dataset JSON file")
    parser.add_argument("--output", required=True, help="Path for output dataset")
    parser.add_argument("--categories", nargs="*", help="Filter by categories (e.g., a b c)")
    parser.add_argument("--complexity", nargs="*", choices=["easy", "medium", "hard"], 
                       help="Filter by complexity levels")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of tasks to process")
    parser.add_argument("--curriculum-order", action="store_true", default=True,
                       help="Sort tasks for curriculum learning progression")
    parser.add_argument("--progressive-batches", action="store_true", 
                       help="Generate progressive batches for curriculum learning")
    parser.add_argument("--batch-size", type=int, default=20, 
                       help="Batch size for progressive generation")
    parser.add_argument("--save-interval", type=int, default=5,
                       help="Save intermediate results every N questions")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Check wakepy availability
    if WAKEPY_AVAILABLE:
        logger.info("🔋 wakepy is available - system will stay awake during generation")
    else:
        logger.warning("⚠️  wakepy not available - consider installing with: pip install wakepy")
    
    try:
        # Initialize generator
        generator = ExtendedDatasetGenerator(args.input)
        await generator.initialize()
        
        if args.progressive_batches:
            # Generate progressive batches
            batch_results = await generator.generate_progressive_batches(
                output_dir=args.output,
                batch_size=args.batch_size,
                complexity_progression=args.complexity
            )
            
            # Print summary
            print("\n🎓 Progressive Batch Generation Summary:")
            for complexity, result in batch_results.items():
                print(f"  • {complexity}: {result['entries_generated']} entries → {result['output_path']}")
        
        else:
            # Generate single dataset with periodic saving
            result = await generator.generate_curriculum_dataset_with_periodic_saving(
                output_path=args.output,
                categories=args.categories,
                complexity_levels=args.complexity,
                curriculum_order=args.curriculum_order,
                max_tasks=args.max_tasks,
                batch_size=args.save_interval
            )
            
            # Print summary
            print(f"\n✅ Dataset generation complete!")
            print(f"  • Generated: {len(result['dataset'])} entries")
            print(f"  • Output: {args.output}")
            print(f"  • Analysis: {args.output.replace('.json', '_analysis.json')}")
            print(f"  • Periodic saving: Every {args.save_interval} questions")
            
            # Print key metrics
            analysis = result["analysis"]
            performance = analysis["performance_summary"]
            print(f"\n📊 Performance Summary:")
            print(f"  • Success Rate: {performance['overall_success_rate']:.1%}")
            print(f"  • Avg Reasoning Score: {performance['avg_reasoning_score']:.1f}")
            print(f"  • Avg Tool Usage: {performance['avg_tool_usage']:.1f}")
            
            # Print recommendations
            recommendations = analysis["recommendations"]
            print(f"\n💡 Training Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"  • {rec}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main()) 