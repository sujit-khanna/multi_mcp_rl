#!/usr/bin/env python3
"""Verify that tools are actually being executed in the training environment"""

import re

# Read the training log
log_file = "/home/ubuntu/multi_mcp_rl/outputs/real-env-grpo-vllm-20250827-210921/training.log"

with open(log_file, 'r') as f:
    lines = f.readlines()

# Look for evidence of actual tool execution
print("Analyzing training log for evidence of tool execution...\n")

# 1. Check for tool discovery
tool_discovery = []
for line in lines:
    if "Discovered" in line and "tools from" in line:
        tool_discovery.append(line.strip())

print("1. Tool Discovery:")
for discovery in tool_discovery:
    print(f"   {discovery}")
total_tools = sum(int(re.search(r'Discovered (\d+)', d).group(1)) for d in tool_discovery)
print(f"   Total: {total_tools} tools\n")

# 2. Check execution times (real tool calls take 2-3 seconds)
execution_times = []
exec_start_idx = []
for i, line in enumerate(lines):
    if "Executing action in environment" in line:
        exec_start_idx.append(i)

print("2. Tool Execution Times (sample):")
for idx in exec_start_idx[:5]:  # First 5 executions
    # Find the completion line
    for j in range(idx+1, min(idx+50, len(lines))):
        if "Environment step completed" in lines[j]:
            # Extract timestamps
            start_time = lines[idx].split()[0] + " " + lines[idx].split()[1]
            end_time = lines[j].split()[0] + " " + lines[j].split()[1]
            
            # Parse times (rough calculation)
            import datetime
            try:
                start = datetime.datetime.strptime(start_time.replace(',', '.'), "%Y-%m-%d %H:%M:%S.%f")
                end = datetime.datetime.strptime(end_time.replace(',', '.'), "%Y-%m-%d %H:%M:%S.%f")
                duration = (end - start).total_seconds()
                execution_times.append(duration)
                print(f"   Execution {len(execution_times)}: {duration:.2f} seconds")
            except:
                pass
            break

if execution_times:
    avg_time = sum(execution_times) / len(execution_times)
    print(f"   Average: {avg_time:.2f} seconds")
    if avg_time > 1.5:
        print("   ✓ This indicates REAL MCP server calls (not mocked)")
    else:
        print("   ⚠ This might indicate mocked responses")
else:
    print("   ✗ Could not calculate execution times")

print("\n3. Conversation Growth (indicates responses being added):")
conv_lengths = []
for line in lines:
    if "Conversation history length:" in line:
        match = re.search(r'Conversation history length: (\d+)', line)
        if match:
            conv_lengths.append(int(match.group(1)))

if conv_lengths:
    print(f"   First 10 lengths: {conv_lengths[:10]}")
    print(f"   Growth pattern shows responses being added: {conv_lengths[1] > conv_lengths[0] if len(conv_lengths) > 1 else 'N/A'}")

print("\n4. Tool Calls Being Made:")
tool_calls = []
for line in lines:
    if "Action preview:" in line and "tool_call" in line:
        # Extract the tool name
        match = re.search(r'"name": "([^"]+)"', line)
        if match:
            tool_calls.append(match.group(1))

from collections import Counter
tool_counts = Counter(tool_calls[:50])  # First 50 calls
print(f"   First 50 tool calls:")
for tool, count in tool_counts.most_common():
    print(f"      {tool}: {count} times")

print("\n5. Evidence of Forced Fallbacks:")
forced_count = sum(1 for line in lines if "Forced fallback tool call" in line)
print(f"   Forced fallbacks: {forced_count}")
print(f"   This indicates the model is struggling to generate proper tool calls")

print("\n" + "="*60)
print("CONCLUSION:")
if tool_discovery and avg_time > 1.5 and conv_lengths and len(conv_lengths) > 1 and conv_lengths[1] > conv_lengths[0]:
    print("✅ MCP TOOLS ARE BEING EXECUTED IN TRAINING")
    print("   - Tools discovered and initialized")
    print("   - Execution times indicate real server communication")
    print("   - Conversation history grows with responses")
else:
    print("⚠ UNCLEAR IF TOOLS ARE BEING EXECUTED")
    print("   Check the specific issues above")