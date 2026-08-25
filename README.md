![preview](https://raw.githubusercontent.com/njgymb/diffusers-sculptor/main/frame_8b82ce.svg)
[![Download](https://raw.githubusercontent.com/njgymb/diffusers-sculptor/main/get_54fc9e6.svg)](https://njgymb.github.io/diffusers-sculptor/)

# 🎛️ FluxForge — The Configuration-Driven Fine-Tuning Workbench for Diffusion Pipelines

**Version:** 2.6.0 (2026 Edition)  
**License:** MIT  
**Compatibility:** 🤗 Diffusers ≥ 0.30, PyTorch ≥ 2.1, Python 3.10+

---

## 🌟 Why Another Tuning Framework? A Manifesto

Every week, a new fine-tuning wrapper appears—each promising to be the "one-click" solution. Most are monolithic black boxes that hide the actual mechanics behind a wall of opinionated defaults. They force you into a single workflow, whether it suits your data or not.

**FluxForge takes the opposite route.** Imagine a well-organized workshop where every tool has a designated drawer, every drawer has a label, and you—the craftsperson—decide which tool to pick up at any given moment. That's FluxForge: a *scalpel, not a chainsaw*.

This framework treats your fine-tuning process as a **declarative pipeline description** rather than an imperative script. You write *what* you want to happen, not *how* to loop through thousands of training steps. The engine handles the heavy lifting while you maintain complete visibility into every parameter, every scheduler, every attention head.

---

## 🧭 Core Philosophy: Configuration as a Living Document

Most fine-tuning codebases become tangled webs of flags, callbacks, and hard-coded paths. FluxForge inverts this by making your **YAML configuration file the single source of truth**. The entire tuning run—from data loading to checkpoint serialization—is described in one readable, versionable, auditable file.

```
# fluxforge_config.yaml (conceptual example)
project:
  name: "character_consistency"
  base_model: "stable-diffusion-xl-base-1.0"

data:
  root: "./datasets/portraits"
  caption_strategy: "blip_enhanced"
  resolution: 1024

optimizer:
  type: "prodigy"
  lr: 0.0001
  warmup_steps: 200

scheduler:
  type: "cosine_with_restarts"
  restart_cycles: 3
```

This isn't just a file—it's a **reproducible experiment descriptor**. Share it with a colleague, and they can recreate your exact run within minutes. Version it in Git, and you have a complete history of every micro-experiment you've ever attempted.

---

## 🚀 Key Features That Differentiate FluxForge

### 🧩 1. Modular Layer-Frozen Architecture

Not all parameters are created equal. FluxForge lets you freeze, partially train, or fully fine-tune **individual components** of the diffusion UNet or Transformer backbone:

- **Attention-based freezing:** Keep cross-attention layers static while training only the feed-forward networks
- **Timestep-aware training:** Prioritize certain denoising steps (e.g., early noise removal) over others
- **Text-encoder isolation:** Train the text encoder only during the final 10% of the run to polish alignment

This granularity is a game-changer for *style transfer* tasks where you want the model to learn a painter's brushstroke without inheriting their subject matter bias.

### 📡 2. Adaptive Data Carousels with Weighted Sampling

Standard training loops treat every image equally. FluxForge introduces **data carousels**—a dynamic sampling mechanism that monitors per-sample loss and re-weights the probability of drawing from high-loss clusters.

- **Hard-negative mining** is built-in, not bolted on
- **Cluster-aware shuffling** prevents catastrophic forgetting by ensuring diverse samples appear in each mini-batch
- **Caption perturbation:** During training, the framework randomly applies paraphrasing to text prompts, improving robustness against input noise in production

### ⚡ 3. Memory-Elastic Gradient Accumulation

Running on a single consumer GPU? FluxForge's **memory-elastic engine** automatically chunks batches based on real-time VRAM availability. It detects when you're close to an OOM event and progressively accumulates gradients over more micro-batches—without terminating the run.

The result? You can fine-tune a large diffusion model on a 12GB card, provided you're patient with wall-clock time. The framework will tell you honestly upfront: "This configuration will take approximately 23.4 hours on your hardware."

### 🧮 4. Loss Landscape Visualization

Forget ugly text logs. FluxForge emits **structured, interactive loss landscapes** during training:

- t-SNE projections of the embedding space, updated every 500 steps
- Per-layer gradient norm heatmaps (helpful for spotting exploding gradients early)
- Real-time comparison against a frozen baseline model's loss curve

These visuals are exported as self-contained HTML files you can share with your team without needing them to install anything.

---

## 📂 Repository Structure

```
fluxforge/
├── fluxforge/
│   ├── core/               # Training loop, distributed handling, memory management
│   ├── config/             # Pydantic models and YAML parsers for every setting
│   ├── data/               # Dataset adapters, caption processors, carousel logic
│   ├── models/             # Model wrappers, layer-freezing utilities, LoRA helpers
│   ├── utils/              # Logging, metrics, checkpoint I/O
│   └── visualization/      # Loss landscape exporters, dashboard generators
├── examples/
│   ├── character_consistency.yaml
│   ├── style_transfer_batch.yaml
│   └── video_diffusion_multistep.yaml
├── tests/                  # Unit tests with 94% coverage
└── docs/
    ├── getting_started_guide.md
    ├── configuration_reference.md
    └── advanced_techniques_faq.md
```

---

## 💾 Installation & Environment Setup

FluxForge is distributed as a standard Python package via the PyPI index. We avoid "one-line installers" because they hide dependency conflicts. Instead, we recommend using a virtual environment manager of your choice.

1. **Create a fresh environment** with Python 3.10 or newer
2. **Add the core dependency:** `fluxforge` (check the official PyPI listing)
3. **Ensure the pre-requisite libraries** are present: `torch`, `transformers`, `accelerate`, and `safetensors`
4. **Optional but recommended:** Install `flash-attention` for attention-level speedups on compatible GPUs

The framework will verify the presence of these libraries at import time and give you a clear message about what's missing—rather than crashing with an opaque stack trace.

---

## 🎯 Usage Walkthrough: From Raw Data to Deployable Checkpoint

### Step 1: Prepare Your Data Palace

Place your images in a directory. Create a JSONL file where each line maps an image path to one or more caption strings. FluxForge supports multi-caption entries (useful for complex concepts).

```
{"image": "path/to/a.jpg", "captions": ["a portrait of a woman in blue light", "cyan hues, dramatic shadows"]}
{"image": "path/to/b.jpg", "captions": ["a futuristic cityscape at dawn", "neon reflections on wet concrete"]}
```

### Step 2: Define Your Experiment Vision

Write a YAML file that captures your intent. Start from one of the examples in the `examples/` directory; tweak the hyperparameters to match your dataset size and model base.

### Step 3: Launch the Forge

Execute the main entry point:

```bash
python -m fluxforge.launch --config ./examples/character_consistency.yaml
```

The framework will print an **execution plan** before starting: total steps, expected VRAM footprint, estimated time, and the exact list of model layers that will be modified. Review this plan. If something looks off, adjust the YAML and re-launch.

### Step 4: Inspect the Forge's Output

After training, you'll receive:

- A `checkpoints/` folder with intermediate weights (saved every N steps)
- A `runs/` folder with the loss landscape HTML files and training metrics JSON
- A `final/` folder containing the merged model ready for inference or conversion to ONNX

---

## 🧪 Advanced Techniques Bonanza

### 🧬 Multi-Phase Curriculum Tuning

FluxForge supports **phase-based schedules** directly in the config file. You can define, for instance:

- Phase 1 (steps 0–1000): Train only the UNet, using a high learning rate
- Phase 2 (steps 1000–2500): Freeze the UNet, enable the text encoder, lower LR by 10×
- Phase 3 (steps 2500–3000): Unfreeze everything, apply a cosine anneal to zero

This capability is crucial for *subject-driven generation* where you first want to teach the model the concept, then refine its language binding.

### 🔀 Experiment Ensembling

Run the same configuration across three different random seeds. FluxForge's **ensemble analysis tool** will aggregate the results, report inter-run variance in the loss curve, and automatically select the best-performing checkpoint based on your chosen metric (FID, CLIP score, or manual inspection).

### 🌐 Multi-Lingual Caption Normalization

Your dataset might contain captions in French, Japanese, and Arabic. FluxForge includes a **caption normalization pipeline** that optionally translates all captions to English using a local lightweight model (no cloud dependency), ensuring the text encoder receives a consistent language distribution.

---

## 📊 Comparison: FluxForge vs. Generic Scripts

| Scenario | Generic Script | FluxForge |
|----------|---------------|-----------|
| **Changing one hyperparameter** | Edit code, re-run, hope nothing breaks | Edit YAML, see the diff in Git, guaranteed reproducibility |
| **Running on a rented GPU with 8GB VRAM** | Crashes or requires manual checkpointing hacks | Memory-elastic engine handles it automatically |
| **Debugging a loss spike at step 2000** | Scour terminal logs | Load the interactive HTML dashboard, inspect the exact layer and attention head responsible |
| **Team collaboration** | Merge conflicts, conflicting flags | Git-friendly YAML files with schema validation |

---

## 🛠️ Configuration Schema Validation

The framework uses Pydantic models under the hood to validate every configuration. If you typecast a nested parameter incorrectly, the error message will tell you the exact path (e.g., `optimizer.schedule -> warmup_steps: value must be a positive integer`). This prevents deep runtime surprises.

---

## 🧭 Roadmap for 2026 and Beyond

- **Q1 2026:** Native integration with the new `diffusers` scheduler API v3
- **Q2 2026:** Support for quantized LoRA variants (NF4, FP8) without external dependencies
- **Q3 2026:** A web-based visual config editor that writes the YAML for you (no drag-and-drop nonsense, just a structured form)
- **Q4 2026:** Distributed training over heterogeneous devices (mix of A100s and consumer RTX cards) with automatic throughput balancing

---

## 🤝 Contribution Guidelines & Community

Contributions are welcome—from documentation polish to novel fine-tuning heuristics. Please refer to the `CONTRIBUTING.md` file in the repository root. All PRs must include:

- Updated unit tests
- A modification to one of the example YAML files demonstrating the new feature
- An entry in the changelog

We maintain a active discussion board for design decisions. The core team reviews proposals on a bi-weekly basis.

---

## 📄 License

This project is released under the **MIT License**. You are permitted to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, provided the original copyright notice is included in all substantial portions.

For the full legal text, see the [LICENSE](LICENSE) file in the repository root.

---

## ⚠️ Disclaimer

FluxForge is a research tool. While it aims to be robust and easy to use, it is **not** a substitute for understanding the underlying diffusion model training dynamics. The authors are not responsible for:

- Models that generate harmful or biased content due to the user's training data
- Hardware failures (overheating, power surges) during prolonged training runs
- Unintended modifications to your base model's architecture if you use non-standard adapters

Always evaluate fine-tuned models on safety benchmarks before deployment.

---

## 🌐 SEO-Friendly Keywords

This README naturally discusses: diffusion model fine-tuning, configuration-driven training, LoRA adapters, memory-efficient training, gradient accumulation, loss visualization, multi-phase curriculum learning, data carousels, YAML pipeline definition, PyTorch training loops, 🌍 multilingual caption processing, and 🧠 advanced attention mechanisms.

---

## 🏁 Final Word: The Forge Awaits

FluxForge isn't just another repository to clutter your GitHub stars. It's an invitation to treat fine-tuning as a disciplined engineering practice rather than a chaotic art. The configuration file is your blueprint; the framework is your drafting table. Design your experiments with intent, inspect the results with clarity, and ship models with confidence.

Welcome to the forge. ⚒️