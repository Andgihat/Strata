<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata-Extended" width="340">
  </picture>
</p>

<h1 align="center">Strata-Extended</h1>
<h3 align="center">Extreme-Efficiency High-Performance Inference Engine for Frontier MoE Models</h3>
<p align="center">Run 125B+ MoE Models with Full 256K Context, Native CP2 Trellis Dequantization, Blackwell sm_120 Acceleration, and DFlash2 Speculation on Consumer 16GB GPUs.</p>

<p align="center">
  <a href="https://github.com/Andgihat/Strata-Extended/releases/latest"><img src="https://img.shields.io/github/v/release/Andgihat/Strata-Extended?label=release&color=0ab887" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Andgihat/Strata-Extended" alt="MIT license"></a>
  <a href="https://github.com/Andgihat/Strata-Extended"><img src="https://img.shields.io/badge/Architecture-Blackwell%20sm__120%20%7C%20Ada%20%7C%20Ampere-orange" alt="Architecture"></a>
  <a href="https://github.com/Andgihat/Strata-Extended"><img src="https://img.shields.io/badge/CP2%20PLE-Native%20147%20TQ2__0-blueviolet" alt="CP2 Format"></a>
</p>

---

## 🏛️ Proof of Capability: 54,000-Voxel Interactive 3D World

Can a **125-billion parameter MoE model** run on a **single consumer GPU with 16 GB VRAM** and write an entire complex, multi-thousand-line 3D interactive application from scratch?

<p align="center">
  <img src="docs/media/sakura-gyro-m-pagoda.png" width="920" alt="Sakura Gyro-M 54,000 Voxel 5-Tier Pagoda generated on RTX 5060 Ti via Strata-Extended">
  <br>
  <em><b>Real generation result:</b> Complete interactive 5-tier Japanese Pagoda (54,054 voxels, Three.js r166, terraced island, torii gates, procedural cherry blossom foliage, wind petal particle physics, day/night lighting, and UI controls). Generated end-to-end by <b>Sakura-Qwen3.8-Flash-Next-Gyro-M (125B MoE)</b> on a single <b>NVIDIA RTX 5060 Ti 16GB</b> using <b>Strata-Extended</b>.</em>
</p>

---

## 🎯 What is Strata-Extended?

**Strata-Extended** is an advanced high-performance fork of the [Strata](https://github.com/Niko1221/Strata) inference engine. While upstream Strata is targeted at standard checkpoints, **Strata-Extended** is built to break memory and format barriers for bleeding-edge community and research models — specifically enabling **gigantic Mixture-of-Experts (MoE) models (100B–125B+)** to execute with full reasoning and deep context on **affordable consumer hardware** (such as the RTX 5060 Ti 16GB, RTX 4070/4080, RTX 3090).

### What Strata-Extended Adds:

1. **Native Sakura CP2 PLE (GGUF Type 147 / `GGML_TYPE_TQ2_0`)**:
   - First-class native C++/CUDA support for the reverse-engineered 42-byte row Trellis quantization format ($160 \times \text{2-bit}$ Trellis codes + FP16 scale $d$).
   - **850x Faster Host LUT**: Precomputed 65,536-entry decoding lookup table (`kSakuraCp2LUT`), reducing CPU dequantization latency from 6,174 ms to **7.05 ms**.
   - **Blackwell (`sm_120`) CUDA Kernel**: Constant-memory codebook batch dequantizer with coalesced `float2` writes, achieving **153.56 GB/s** memory throughput (0.518 ms for 124K tokens).
   - **Direct I/O SSD Streaming**: Bypasses OS file cache for the 12.5 GB CP2 n-gram tables via `platform::DirectFile` (`FILE_FLAG_NO_BUFFERING`).

2. **DFlash2 Native Head Speculative Decoding**:
   - `STRATA_DF2_HEAD=native`: Projects drafts directly against the target model's full **248,320-token vocabulary** in VRAM, eliminating token dropping in non-English languages (Russian, CJK) and complex code syntax present in the pruned 106K draft vocabulary.
   - **Full Speculation Window (`--spec 7 --draft-cap 7`)**: Unlocks up to **3.21 accepted tokens per decode round** (50–85 tok/s).

3. **Fused TQ Prefill**:
   - `STRATA_PF_FUSED_TQ=1`: Single-pass fused Hadamard rotation and matrix multiplication kernels on Blackwell/Ada (`sm_80+`), achieving **655–933+ tok/s** prefill speed with zero precision loss.

4. **Adaptive Tail-Skip MoE Routing (0 / 5 / 7)**:
   - **Tail-Skip 0**: Pure 100% MoE evaluation (all 10 experts per token).
   - **Tail-Skip 5 (Balance)**: Drops the lowest 5 tail experts (~10% probability mass), saving 50% MoE FFN compute while retaining 90% model capacity.
   - **Tail-Skip 7 (Ultra-Turbo)**: Drops 7 tail experts, evaluating Top-3 active experts. Slashes SSD expert swaps by >70% and drives sustained speeds of **55–65+ tok/s**, proven capable of generating 54K-voxel Three.js applications without losing spatial or syntactic integrity.

5. **3-Tier Sliding Window Memory (Full 256K Context)**:
   - Full **262,144-token context window** with a **32,768-token VRAM sliding window** (`--kv-resident 32768`).
   - Retains 18.3 GB RAM VirtualLock arena with 0.3 GiB headroom, minimizing SSD I/O to negligible amounts.

6. **CPU Multimodal Vision (`strata-vision`)**:
   - Offloads visual image encoding (e.g. `mmproj-Qwen3.8-Flash-Next-BF16.gguf`) to 6 host CPU threads, reserving all 16 GB of VRAM for the MoE expert arena and KV cache.

7. **Drop-in OpenAI & Anthropic HTTP Server**:
   - Full compatibility with **OpenCode**, **OpenChamber**, **LibreChat**, **Continue**, and terminal coding agents on `http://127.0.0.1:8080/v1`.

---

## 📊 Performance Benchmarks

*Tested on: AMD Ryzen 5 9600X, 32GB DDR5-6000 RAM, NVIDIA GeForce RTX 5060 Ti 16GB, PCIe 4.0 NVMe SSD.*

| Engine / Mode | Context Window | Prefill Speed | Decode Speed | Speculation Acceptance | SSD I/O per Round |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla Engine (No CP2/Spec)** | 256K | 412 tok/s | 23.5 tok/s | 1.00 (No Draft) | 4.3 MB |
| **Strata-Extended [Pure Quality (TS 0)]** | 256K | **655+ tok/s** | **45.0 tok/s** | **3.05 tok/round** | **0.61 MB** |
| **Strata-Extended [Balance (TS 5)]** | 256K | **655+ tok/s** | **49.6 tok/s** | **3.12 tok/round** | **0.42 MB** |
| **Strata-Extended [Ultra-Turbo (TS 7)]** | 256K | **655+ tok/s** | **58–65+ tok/s** | **3.21 tok/round** | **<0.15 MB** |
| **Zero-Disk in RAM Mode** | 32K | **933+ tok/s** | **65.5 tok/s** | **3.21 tok/round** | **0.00 MB** |

---

## 🚀 Quick Start

### 1. Download Release
Download the prebuilt Windows CUDA bundle from [Releases](https://github.com/Andgihat/Strata-Extended/releases/latest):
* `strata-sakura-cp2-v0.2.16-windows-x86_64-cuda.zip`

### 2. Launching with the Interactive Launcher
Extract the archive and run:
```cmd
start_sakura_gyro_m.cmd
```
You will be prompted to select your execution profile:
* `[1] DFlash2 + Pure Quality  [No Tail-Skip, 100% MoE, ~45 t/s, PP 655+]`
* `[2] DFlash2 + Balanced      [Tail-Skip 5, speedup up to ~50 t/s, PP 655+]`
* `[3] DFlash2 + Ultra-Turbo   [Tail-Skip 7, speedup up to ~65+ t/s, verified on 3D code]` *(Recommended)*
* `[4] Pure Decode             [100% MoE, without DFlash2 speculation, ~23.5 t/s]`

### 3. Connecting to OpenCode
Add the provider to `~/.config/opencode/opencode.json`:
```json
"s2": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "S2 Sakura 125B",
  "options": {
    "baseURL": "http://127.0.0.1:8080/v1"
  },
  "models": {
    "sakura": {
      "name": "[xhigh] Sakura Gyro-M 256K (Max reasoning)",
      "limit": { "context": 262144, "output": 32768 },
      "options": { "reasoning": { "effort": "xhigh" } }
    }
  }
}
```

---

## 🛠️ Building from Source

```bash
git clone https://github.com/Andgihat/Strata-Extended.git
cd Strata-Extended
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSTRATA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120"
cmake --build . --config Release --parallel
```

---

## 📄 License & Credits
Licensed under the [MIT License](LICENSE).  
Based on [Strata](https://github.com/Niko1221/Strata) by Niko Veit.  
Sakura CP2 architecture, Trellis kernels, Blackwell optimizations, and extended serving pipeline developed by [Andgihat](https://github.com/Andgihat).
