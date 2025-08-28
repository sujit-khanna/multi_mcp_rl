## Multi‑MCP RL: vLLM Real‑Env GRPO Analysis

Scope and entry point
- Command: `ENABLE_VLLM=true VLLM_MAX_MODEL_LEN=4096 VLLM_GPU_MEMORY_UTILIZATION=0.3 ./training/scripts/launch_real_env_gpu_vllm.sh`
- Script calls: `training/scripts/train_qwen3_grpo_real_env_vllm.py` → uses `training/core/{grpo_trainer.py, grpo_trainer_with_value.py, grpo_trainer_gradient_fix.py}`, `training/data/trajectory_collector.py`, `environments/{mcp_tool_environment.py, simple_shared_manager.py}`, and `mcp_tools/limited/*`.
- Logs reviewed: `outputs/real-env-grpo-vllm-20250824-145444/training.log`.

Key findings (root causes and evidence)
- Tool execution never triggers: Policy outputs natural language, not `<tool_call>…</tool_call>`. Log shows `tool_calls_this_turn: 0` consistently. Reward remains ~0; training signal is weak.
- Env/collector contract mismatch: `MCPToolEnvironment.step()` returns key `observations` (list of messages), but `TrajectoryCollector` expects `observation` (string). Result: no observation appended to history, reducing state fidelity for next turns.
- Tools never initialized in env: Collector sets `env.tool_manager = SimpleSharedMCPToolManager()` but never calls `await env.initialize_tools()`. `MCPToolEnvironment` then checks `tool_name in self.available_tools` (empty), so even correct tool calls would be rejected.
- Trajectory state snapshots missing: Collector’s per‑turn record lacks the pre‑action state (`state`). Later, training builds trajectories with `step.get('state', {})`, yielding `{}`; `compute_log_probs/compute_values` then condition on the wrong prompt.
- Prompt/template inconsistency: Sampling uses a Qwen chat template with `<|im_start|>…` while log‑prob/value code builds prompts as `role: content` text. PPO ratios and values are computed against a different conditioning than used at sampling → inflated KL and incorrect gradients. Logs show repeated “High KL divergence” warnings.
- GRPO grouping ineffective: Training loop samples distinct tasks (`batch_size` tasks) but GRPO expects multiple rollouts per task (`group_size`) for relative rewards. Current setup rarely forms groups, weakening the objective.
- Dataset/tools drift: Logs include intents like `tavily_filter` which is not provided by `mcp_tools/limited` (only `tavily_search`, `tavily_qna_search`, etc.). Mismatched tool names guarantee zero valid tool calls.

Recommended fixes (actionable)
- Enforce tool‑call generation: Use `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` for Qwen; set `stop=["</tool_call>"]`, `min_tokens>=8`, lower `temperature` (e.g., 0.2–0.4) for early phases; add rejection sampling if no tool call found.
- Unify prompting across sampling/log‑prob/value: Centralize a prompt builder and use it in `generate_action`, `compute_log_probs`, and `compute_values` to ensure identical conditioning and tokenization.
- Initialize tools and fix step I/O: In collector, call `await env.initialize_tools()` after env creation; in env.step, either add `"observation"` key mirroring the current `observations` or update collector to read `observations` and convert to a string/user message consistently.
- Record correct per‑turn state: Before generating an action, deep‑copy the current conversation history and store it as `turn_data['state']`; train on that state.
- Honor GRPO group size: Sample K tasks and collect `group_size` episodes per task (e.g., resample with different seeds) before `train_step`.
- Reference policy: You already clone a non‑quantized reference; remove stale log lines claiming “KL divergence is 0 (same policy)” and verify ref updates happen (`ref_policy_update_frequency≈100`). Consider scaling KL by tokens.
- Align dataset tool names with `mcp_tools/limited` or add missing tools; validate via a pre‑run check.

Validation checklist
- Unit sanity: one rollout produces a valid `<tool_call>`; env executes tool; `tools_used>0`; reward > 0.
- PPO signals: non‑zero old/current log‑probs; KL in a reasonable band (< 5e‑1 after warmup); advantages finite; gradient norm bounded.
- Groups form: each `task_id` has ≥ `group_size` trajectories per update.

