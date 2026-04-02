# Red Team Test Results

> **WARNING: ACADEMIC RED-TEAM RESEARCH ONLY.**
> This repository is provided solely for lawful security research and evaluation.
> Do **not** use it for illegal, harmful, or unauthorized activity.
> Use of this repository is at your own risk.
> The author provides no warranty, makes no endorsement of downstream use, and assumes no liability for misuse or resulting consequences.
> If you are unsure whether your intended use is permitted, consult qualified legal counsel before proceeding.
> This repository is only a scaffold for organizing red-team test results.
> The actual method code is not provided in this repository.
> The datasets are not provided in this repository either.

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

## References

- `Crescendo`: https://github.com/Azure/PyRIT/tree/main, https://crescendo-the-multiturn-jailbreak.github.io/assets/pdf/CrescendoFullPaper.pdf
- `TAP`: https://github.com/marcellopoliti/tree-of-attacks, https://arxiv.org/pdf/2312.02119
- `CKA-Agent`: https://github.com/Graph-COM/CKA-Agent, https://arxiv.org/pdf/2512.01353
- `x-teaming`: https://github.com/salman-lui/x-teaming, https://arxiv.org/abs/2504.13203, https://x-teaming.github.io/
