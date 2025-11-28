# Mamba-MoE 300M — Hybrid Language Model

A cutting-edge 300M parameter language model combining Mamba 2 state-space models, Soft MoE, and hierarchical memory for exceptional chat performance on mobile devices.

## 🎯 Project Goals

- **Performance**: Match 1B+ models with only 300M parameters
- **Memory**: Best-in-class context retention with 4-tier hierarchical memory
- **Mobile**: Run at 15+ tok/s on mid-range Android devices (Realme 6)
- **Modular**: Support post-training expansion (experts, LoRA, layers)

## 🏗️ Architecture Highlights

- **Mamba 2 SSM Backbone**: Linear-time long context (24 layers, 1024 dim)
- **Soft MoE**: 4-6 domain experts with shared expert (stable training)
- **Differential Multi-Query Attention**: Enhanced long-context accuracy
- **Factorized Embeddings**: 32k vocab, parameter-efficient
- **YaRN RoPE**: Train 8k, infer 32k+ context
- **4-Tier Memory**: Working → Conversation → Long-term → External

## 📊 Target Benchmarks

| Benchmark | Target | Baseline (Phi-3-mini 3.8B) |
|-----------|--------|---------------------------|
| MT-Bench | >7.0 | 8.0 |
| GSM8K | >40% | 82% |
| HumanEval | >30% | 58% |
| Memory Recall | >80% | N/A |

## 🛠️ Tech Stack

- **Framework**: JAX + Flax
- **Hardware**: TPU v3-8 (Kaggle/Colab)
- **Training**: Teacher-guided distillation (Qwen 2.5 7B)
- **Data**: 1.5TB+ (FineWeb-Edu, Stack, OpenWebMath, etc.)
- **Export**: GGUF + llama.cpp for mobile

## 📁 Project Structure

```
mamba-moe-300m/
├── src/
│   ├── model/          # Model architecture (Mamba, MoE, Attention, Memory)
│   ├── training/       # Training loop, optimization, distillation
│   ├── data/           # Data loading, preprocessing, streaming
│   ├── evaluation/     # Benchmarks, metrics, testing
│   └── export/         # GGUF conversion, quantization
├── configs/            # Model & training configs (YAML)
├── scripts/            # Utility scripts (data download, setup, etc.)
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks for analysis
├── data/               # Raw & processed data (not in git)
├── checkpoints/        # Model checkpoints (not in git)
└── logs/               # Training logs
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd mamba-moe-300m

# Install dependencies
pip install -r requirements.txt

# Test TPU connection
python scripts/test_tpu.py
```

### 2. Prepare Data

```bash
# Download datasets (requires ~2TB storage)
python scripts/download_data.py --datasets fineweb,stack,openwebmath

# Preprocess & tokenize
python scripts/preprocess_data.py --output data/processed/
```

### 3. Train Model

```bash
# Stage 1: Base pretraining
python -m src.training.train \
    --config configs/stage1_pretrain.yaml \
    --tpu v3-8

# Stage 2: Instruction tuning
python -m src.training.train \
    --config configs/stage2_instruction.yaml \
    --checkpoint checkpoints/stage1_final/

# Stage 3-5: Reasoning, Memory, QAT
# (see training guide)
```

### 4. Evaluate

```bash
# Run benchmarks
python -m src.evaluation.run_benchmarks \
    --checkpoint checkpoints/stage5_final/ \
    --output results/
```

### 5. Export for Mobile

```bash
# Convert to GGUF & quantize
python -m src.export.to_gguf \
    --checkpoint checkpoints/stage5_final/ \
    --quantize q4_k_m \
    --output models/mamba_moe_300m_q4.gguf
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Developer Guide](docs/development.md)

## 📈 Training Timeline

| Phase | Duration | Hardware |
|-------|----------|----------|
| Data prep | 1-2 weeks | CPU |
| Stage 1 (Pretrain) | 3-5 days | TPU v3-8 |
| Stage 2-4 (Tuning) | 3-4 days | TPU v3-8 |
| Stage 5 (QAT) | 1 day | TPU v3-8 |
| Export & test | 2-3 days | CPU/GPU |
| **Total** | **2-3 weeks** | (with free TPU) |

## 🎓 Citation

If you use this model, please cite:

```bibtex
@software{mamba_moe_300m,
  title={Mamba-MoE 300M: A Modular Hybrid Language Model},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/mamba-moe-300m}
}
```

## 📝 License

MIT License (see LICENSE file)

## 🙏 Acknowledgments

- Mamba 2 architecture: [Dao & Gu 2024]
- Soft MoE: [Puigcerver et al. 2023]
- Differential Attention: [Microsoft Research 2024]
- Teacher models: Qwen, DeepSeek teams

---

**Status**: 🚧 In Development

**Version**: v0.1.0-alpha

**Author**: Prince

**Contact**: [Your contact info]
