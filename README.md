# Awesome LLM Kernel Agent

## Methods
|  Title  |   Venue  |   Date   |   Code   |   topic   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| [**CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization**](https://arxiv.org/abs/2511.01884) <br> | arXiv | 2025.11 | [Github](https://github.com/OptimAI-Lab/CudaForge) | CUDA, Agent, Iterative Search |
| [**STARK: Strategic Team of Agents for Refining Kernels**](https://arxiv.org/abs/2510.16996) <br> | arXiv | 2025.10 | - | CUDA, Multi-Agent, Iterative Search |
| [**From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph**](https://arxiv.org/abs/2510.19873) <br> | arXiv | 2025.10 | - | CUDA, RAG, Transfer Learning |
| [**ConCuR: Conciseness Makes State-of-the-Art Kernel Generation**](https://arxiv.org/abs/2510.07356) <br> | arXiv | 2025.10 | - | CUDA, Finetuning |
| [**Mastering Sparse CUDA Generation through Pretrained Models and Deep Reinforcement Learning**](https://openreview.net/forum?id=VdLEaGPYWT) <br> | ICLR 2026 | 2025.09 | - | CUDA, Sparse, RL, Finetuning |
| [Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization](https://arxiv.org/abs/2509.14279) <br> | arXiv | 2025.09 | [Github](https://github.com/SakanaAI/robust-kbench) | CUDA |
| [Astra: A Multi-Agent System for GPU Kernel Performance Optimization](https://arxiv.org/abs/2509.07506) | arXiv | 2025.09 | - | CUDA |
| [SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization](https://arxiv.org/abs/2508.20258) | arXiv | 2025.08 | - | AMD GPU |
| [CudaLLM: Training Language Models to Generate High-Performance CUDA Kernels](https://huggingface.co/ByteDance-Seed/cudaLLM-8B) <br> | HugginFace | 2025.08 | [Github](https://github.com/ByteDance-Seed/cudaLLM) | CUDA, Finetuning |
| [OS-R1: Agentic Operating System Kernel Tuning with Reinforcement Learning](https://arxiv.org/abs/2508.12551) <br> | arXiv | 2025.08 | - | General Kernel, OS Kernel |
| ![Star](https://img.shields.io/github/stars/AMD-AIG-AIMA/GEAK-agent.svg?style=social&label=Star) <br> [**GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks**](https://arxiv.org/abs/2507.23194) <br> | arXiv | 2025.07 | [Github](https://github.com/AMD-AIG-AIMA/GEAK-agent) | Triton (AMD), Iterative Search |
| [**AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs**](https://arxiv.org/abs/2507.05687) <br> | arXiv | 2025.07 | [Github](https://github.com/AI9Stars/AutoTriton) | Triton, Finetuning, RL |
| [**GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**](https://arxiv.org/abs/2507.19457) <br> | arXiv | 2025.07 | - | AMD NPU + CUDA, Prompt Engineering |
| ![Star](https://img.shields.io/github/stars/deepreinforce-ai/CUDA-L1.svg?style=social&label=Star) <br> [**CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning**](https://arxiv.org/abs/2507.14111) <br> | arXiv | 2025.07 | [Github](https://github.com/deepreinforce-ai/CUDA-L1) | CUDA, Finetuning |
| [**Omniwise: Predicting GPU Kernels Performance with LLMs**](https://arxiv.org/abs/2506.20886) <br> | arXiv | 2025.06 | - | AMD NPU, Surrogate Modeling |
| [**Kevin: Multi-Turn RL for Generating CUDA Kernels**](https://openreview.net/forum?id=HLeyRyV55o) <br> | EXAIT Workshop @ ICML | 2025.06 | - | CUDA, Finetuning |
| [**GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization**](https://openreview.net/forum?id=K4XSvet59a) <br> | ES-FoMo Workshop @ ICML | 2025.06 | - | AMD, Iterative Search |
| [**CUDA-LLM: LLMs Can Write Efficient CUDA Kernels**](https://arxiv.org/abs/2506.09092) <br> | arXiv | 2025.06 | - | CUDA, Prompt Engineering (?) |
| [**The AI CUDA Engineer: Agentic CUDA Kernel Discovery, Optimization and Composition**](https://pub.sakana.ai/ai-cuda-engineer) <br> | arXiv | 2025.02 | [HuggingFace](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive) | CUDA, Agent + Iterative Search + RAG |

## Datasets and Benchmarks
|  Title  |   Venue  |   Date   |   Code   |   topic   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/wzzll123/MultiKernelBench.svg?style=social&label=Star) <br> [**MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation**](https://www.arxiv.org/abs/2507.17773) <br> | arXiv | 2025.07 | [Github](https://github.com/wzzll123/MultiKernelBench) | CUDA + CANN + TPU |
| [**NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers**](https://arxiv.org/abs/2507.14403) <br> | arXiv | 2025.07 | - | NPU |
| ![Star](https://img.shields.io/github/stars/thunlp/TritonBench.svg?style=social&label=Star) <br> [**TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators**](https://arxiv.org/abs/2502.14752) <br> | ICML | 2025.02 | [Github](https://github.com/thunlp/TritonBench) | Triton (CUDA) |
| ![Star](https://img.shields.io/github/stars/ScalingIntelligence/KernelBench.svg?style=social&label=Star) <br> [**KernelBench: Can LLMs Write Efficient GPU Kernels?**](https://arxiv.org/abs/2502.10517) <br> | ICML | 2025.02 | [Github](https://github.com/ScalingIntelligence/KernelBench) | CUDA |
| [**Comparing Llama-2 and GPT-3 LLMs for HPC kernels generation**](https://arxiv.org/abs/2309.07103) <br> | [LCPC](http://www.lcpcworkshop.org/LCPC23/) | 2023.09 | - |  |

## Others
|  Title  |   Date   |   topic   |
|:--------|:--------:|:--------:|
|[AMD Developer Challenge 2025](https://www.datamonsters.com/amd-developer-challenge-2025)| 2025.04 | AMD GPU |
|[GPU Mode Learderboard](https://www.gpumode.com/)| - | AMD + CUDA |