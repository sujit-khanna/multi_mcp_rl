# Repository Guidelines

## Project Structure & Module Organization
- `training/`: core training code (`core/`, `data/`, `utils/`, `scripts/`, `configs/`).
- `environments/`: MCP tool environment adapters and shared tool managers.
- `mcp_tools/limited/`: local MCP servers used during training/eval.
- `agents/`: example trajectories and agent utilities.
- `tests` and `training/tests`: PyTest suites and utilities.
- `configs/`, `data/`, `outputs/`: config stubs, datasets, and run artifacts (git‑ignored).

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt` (Python 3.12+).
- Activate env + vars: `source setup_env.sh` (sets `PYTHONPATH`, loads `.env`).
- Run unit tests: `pytest -q` or `pytest training/tests -q`.
- Quick local run (CPU): `./training/scripts/launch_real_env_cpu.sh`.
- Single‑GPU LoRA: `./training/scripts/launch_qwen3_training.sh`.
- Multi‑GPU (DeepSpeed): `./training/scripts/launch_distributed.sh`.

## Coding Style & Naming Conventions
- Python, PEP8, 4‑space indentation; prefer type hints for public APIs.
- Filenames: `snake_case.py`; Classes: `PascalCase`; functions/vars: `snake_case`.
- Docstrings: short, action‑oriented; include shapes/types for tensors.
- Async code: use `async/await` for tool I/O and rollouts; avoid blocking calls.

## Testing Guidelines
- Framework: PyTest (+ `pytest-asyncio`).
- Name tests `test_*.py`; colocate new tests under `training/tests/` when touching training, or root for cross‑cutting tests.
- Run subsets: `pytest -k mcp_integration -q`.
- Aim for coverage of: environment interactions, trajectory collection, trainer steps, and failure modes (e.g., tool timeouts, CUDA OOM guards).

## Commit & Pull Request Guidelines
- History is mixed; use clear, imperative messages. Prefer Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Keep PRs focused; include: purpose, key changes, config impacts, and test evidence (logs/screenshots from `outputs/` or WandB run link).
- Link related issues; note any schema/config changes under `training/configs/`.

## Security & Configuration Tips
- Never commit secrets. Keep `.env` local; reference required keys in README.
- Validate local config with a dry run on CPU (`launch_real_env_cpu.sh`).
- When adding tools, place servers in `mcp_tools/limited/` and register via `environments/simple_shared_manager.py`; include a minimal integration test.

