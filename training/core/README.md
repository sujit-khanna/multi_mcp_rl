# Environment Adapter Integration Layer

This module provides seamless integration between the existing `MCPToolEnvironment` and GRPO training infrastructure.

## ✅ **Integration Complete**

All integration tests pass (4/4):
- ✅ Basic Integration
- ✅ Policy Integration  
- ✅ Training Iterator
- ✅ Async Bridge

## 🔧 **Key Components**

### 1. **EnvironmentAdapter**
Main integration class that bridges MCPToolEnvironment with training policies.

```python
from training.core import create_environment_adapter

# Create adapter with shared tool manager
adapter = create_environment_adapter(
    data_path="data/processed/train.json",
    use_shared_tools=True
)
```

### 2. **PolicyInterface**
Standard interface for external policies to work with the environment.

```python
from training.core import PolicyInterface

# Define your policy
def my_policy(conversation: List[Dict[str, str]]) -> str:
    # Generate action with reasoning and tool calls
    return "<think>...</think>\n<tool_call>...</tool_call>"

# Wrap in interface
policy = PolicyInterface(
    generate_action=my_policy,
    reset=lambda: None  # Optional reset function
)
```

### 3. **SharedToolManagerEnvironment**
Extended MCPToolEnvironment that uses a shared tool manager across all environments.
- Prevents creating new MCP connections for each environment
- Critical for training efficiency with 100k+ episodes

### 4. **AsyncToSyncBridge**
Handles async MCP tool operations in synchronous training loops.
- Dedicated event loop in separate thread
- Compatible with PyTorch training loops

### 5. **TaskDataLoader**
Loads and formats tasks from `data/processed/train.json`.
- Validates task structure
- Adds missing fields with defaults
- Provides batch and iterator interfaces

## 📊 **Usage Example**

```python
from training.core import create_environment_adapter, PolicyInterface

# 1. Create adapter
adapter = create_environment_adapter()

# 2. Define your training policy
class MyTrainingPolicy:
    def generate_action(self, conversation):
        # Your policy logic here
        return action_string

# 3. Create training iterator
policy = PolicyInterface(generate_action=MyTrainingPolicy().generate_action)
iterator = adapter.create_training_iterator(
    policy_fn=policy,
    batch_size=32,
    max_episodes_per_task=1,
    shuffle=True
)

# 4. Training loop
for batch in iterator:
    episodes = batch["episodes"]
    for episode in episodes:
        # Access trajectory data
        observations = episode["observations"]
        actions = episode["actions"]
        rewards = episode["rewards"]
        
        # Train your model...
```

## 🎯 **Key Features**

### **Removes PolicyIntegratedEnvironment Dependency**
- External policies can be plugged in directly
- No need to modify environment code

### **Shared Tool Manager**
- Single tool manager instance across all environments
- Prevents connection exhaustion with 100k+ episodes
- ~23 tools available from 5 MCP servers

### **Async Tool Execution**
- Seamless async-to-sync conversion
- Compatible with standard PyTorch training loops
- Thread-safe execution

### **Task Data Compatibility**
- Loads tasks from `data/processed/train.json`
- Validates and formats task structure
- Handles missing fields gracefully

## 🚀 **Performance Characteristics**

- **Tool Initialization**: ~2 seconds (one-time with shared manager)
- **Episode Generation**: Fast with pre-initialized tools
- **Memory Efficient**: Shared resources across environments
- **Thread Safe**: Async bridge handles concurrent operations

## 🔌 **Integration Points**

This adapter integrates with:
- ✅ `MCPToolEnvironment` - Existing environment code
- ✅ `RealMCPToolManager` - Real tool execution
- ✅ Training data format - `data/processed/train.json`
- ✅ GRPO training loop - External policy support

The integration is complete and ready for GRPO training implementation!