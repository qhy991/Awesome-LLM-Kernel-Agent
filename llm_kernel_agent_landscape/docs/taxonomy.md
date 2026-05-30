# Taxonomy

This taxonomy is designed for maintaining a landscape map of LLM-driven kernel generation, kernel optimization and kernel engineering agents.

## Categories

### LLM4Kernel

Works that primarily improve the model's ability to generate or optimize kernels.

Typical signals:

- CUDA / Triton / HIP specific SFT
- RL or contrastive RL for kernel optimization
- model specialization for GPU kernels
- reward design for speedup and correctness

### Agent4Kernel

Works that organize kernel engineering as an agentic workflow.

Typical signals:

- planning + tool use
- compile / run / benchmark feedback
- profiling-guided optimization
- multi-agent coder / judge / verifier workflows
- memory, experience or skill management for agents

### Datasets

Training, retrieval, code or knowledge resources that support kernel generation and optimization.

Typical signals:

- structured instruction data
- PyTorch-to-CUDA / PyTorch-to-Triton pairs
- operator libraries and kernel libraries
- code repositories / DSL examples
- benchmark traces or profiling traces

### Benchmarks

Evaluation suites that measure correctness, speedup, robustness or backend generalization.

Typical signals:

- functional correctness tests
- PyTorch baseline speedup
- Triton/CUDA/HIP/Metal/NPU/TPU coverage
- robustness and anti-cheating checks
- end-to-end agent evaluation

### Systems / Platforms

Integrated systems that combine generation, evaluation, experience, skills, training and feedback.

Typical signals:

- end-to-end kernel engineering workflows
- policy registry / canary / rollback
- multi-backend verification
- self-evolution or long-term experience accumulation

KernelOwl belongs here, while also connecting to the other four categories through KernelTrain, KernelEval, KernelSkill and EvolutionLedger.

## Periods

The default landscape uses coarse time bands:

- `2024`
- `2025 H1`
- `2025 H2`
- `2026+`

Use publication date, preprint date, or first public release date. When uncertain, use the most conservative period and add a note.

## Inclusion criteria

Core inclusion:

- LLM directly generates or optimizes CUDA / Triton / HIP / Metal / NPU / TPU kernels.
- Agentic workflows for kernel engineering.
- Kernel-specific SFT / RL / reward design.
- Kernel generation benchmarks or datasets.
- Memory / skill / experience systems for kernel agents.

Extended inclusion:

- HPC code generation with direct relevance to kernels.
- Operator/kernel libraries useful as training or retrieval sources.
- DSL and compilation frameworks that are used by LLM-driven kernel systems.

Usually exclude:

- generic code generation without kernel-specific evaluation.
- generic software engineering agents without performance-critical kernel tasks.
- generic AI infrastructure work with no kernel engineering component.
