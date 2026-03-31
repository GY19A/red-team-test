# Red Team Test Results

This repository stores red-team evaluation results for four jailbreak methods:

- `Crescendo`
- `TAP`
- `CKA-Agent`
- `x-teaming`

## Model Setup

The experiments use the model configuration from `configs/models.yaml` in the main project:

- Attacker model: `Qwen/Qwen3-32B`
- Defender model: `openai/gpt-oss-120b`

In the config file, these correspond to:

- `attackers.qwen-3-32B`
- `defenders.gpt-oss-120B`

The attacker model generates jailbreak attempts, and the defender model is the target model being tested against those attacks.

## Benchmark Setting

The evaluation is run on two benchmarks:

- `JailbreakBench`
- `HarmBench`

For each benchmark, the evaluation uses a prepared 50-sample subset for testing.

That means each run evaluates:

- 50 samples from `JailbreakBench`
- 50 samples from `HarmBench`

for a total of 100 prompts.

## Repository Layout

The results are organized by method under `results/`.

Example structure:

```text
results/
  CKA-Agent/
  Crescendo/
  TAP/
  x-teaming/
```
