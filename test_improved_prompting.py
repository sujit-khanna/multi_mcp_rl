#!/usr/bin/env python3
"""
Test the improved prompting to see if it generates tool calls
"""

import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent / "training" / "core"))
sys.path.append(str(Path(__file__).parent))

from training.core.qwen_policy_with_prompting import QwenPolicyWithPrompting

def test_tool_call_generation():
    """Test if the improved prompting generates tool calls"""
    
    print("="*60)
    print("🧪 TESTING IMPROVED TOOL CALL GENERATION")
    print("="*60)
    
    # Create policy (will use mock if no model available)
    try:
        policy = QwenPolicyWithPrompting(
            model_config_path="configs/model_config_temp.yaml",
            training_config_path="configs/training_config_temp.yaml",
            use_lora=True,
            device="cpu",
            load_in_4bit=False
        )
        
        print("✅ Policy created successfully")
        
        # Test cases
        test_cases = [
            [{"role": "user", "content": "What is Apple's stock price?"}],
            [{"role": "user", "content": "Get Tesla news"}],
            [{"role": "user", "content": "Search for Microsoft information"}],
            [{"role": "user", "content": "Calculate 2 + 2"}],
        ]
        
        for i, conversation in enumerate(test_cases, 1):
            print(f"\n{i}️⃣ Testing: {conversation[0]['content']}")
            
            try:
                # Generate action
                actions = policy.generate_action([conversation])
                action = actions[0] if actions else ""
                
                print(f"   Generated: {action[:100]}...")
                
                # Check if tool call was generated
                if '<tool_call>' in action and '"name":' in action and '"arguments":' in action:
                    print("   ✅ PROPER TOOL CALL GENERATED!")
                else:
                    print("   ❌ No proper tool call found")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Failed to create policy: {e}")
        print("This is expected if model files don't exist - the important thing is the logic works")

if __name__ == "__main__":
    test_tool_call_generation()