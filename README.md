# MoviiGen 1.1

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/ZuluVision/MoviiGen1.1)
[![GitHub stars](https://img.shields.io/github/stars/ZulutionAI/MoviiGen1.1?style=social)](https://github.com/ZulutionAI/MoviiGen1.1/stargazers)


[**MoviiGen 1.1: Towards Cinematic-Quality Video Generative Models**]("https://huggingface.co/ZuluVision/MoviiGen1.1") <be>

In this repository, we present **MoviiGen 1.1**, a cutting-edge video generation model that excels in cinematic aesthetics and visual quality. This model is a fine-tuning model based on the Wan2.1. Based on comprehensive evaluations by 11 professional filmmakers and AIGC creators, including industry experts, across 60 aesthetic dimensions, **MoviiGen 1.1** demonstrates superior performance in key cinematic aspects:

- 👍 **Superior Cinematic Aesthetics**: **MoviiGen 1.1** outperforms competitors in three critical dimensions: atmosphere creation, camera movement, and object detail preservation, making it the preferred choice for professional cinematic applications. 
- 👍 **Visual Coherence & Quality**: MoviiGen 1.1 excels in clarity (+14.6%) and realism (+4.3%), making it ideal for high-fidelity scenarios such as real-scene conversion and portrait detail. Wan2.1 stands out in smoothness and overall visual harmony, better suited for tasks emphasizing composition, coherence, and artistic style. Both models have close overall scores, so users can select MoviiGen 1.1 for clarity and realism, or Wan2.1 for style and structural consistency.
- 👍 **Comprehensive Visual Capabilities**: **MoviiGen 1.1** provides stable performance in complex visual scenarios, ensuring consistent subject and scene representation while maintaining high-quality motion dynamics.
- 👍 **High-Quality Output**: The model generates videos with exceptional clarity and detail, supporting both 720P and 1080P resolutions while maintaining consistent visual quality throughout the sequence.
- 👍 **Professional-Grade Results**: **MoviiGen 1.1** is particularly well-suited for applications where cinematic quality, visual coherence, and aesthetic excellence are paramount, offering superior overall quality compared to other models.

This repository features our latest model, which establishes new benchmarks in cinematic video generation. Through extensive evaluation by industry professionals, it has demonstrated exceptional capabilities in creating high-quality visuals with natural motion dynamics and consistent aesthetic quality, making it an ideal choice for professional video production and creative applications.

## Video Demos

| <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/79_1920*1056_seed3732225395.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/150_1920*1056_seed1674457713.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/143_1920*1056_seed3114534932.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/94_1920*1056_seed3693446494.mp4" type="video/mp4">Your browser does not support the video tag.</video> |
|--------|--------|--------|--------|
| <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/23_1920*1056_seed3934691816.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/13_1920*1056..mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/26_1920*1056..mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/39_1920*1056..mp4" type="video/mp4">Your browser does not support the video tag.</video> |
| <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/100_1920*1056_seed2949593166.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/54_1920*1056..mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/107_1920*1056_seed525896597.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="https://huggingface.co/ZuluVision/MoviiGen1.1/resolve/main/assets/163_1920*1056_seed3696194034.mp4" type="video/mp4">Your browser does not support the video tag.</video> |


## 🔥 Latest News!!

* May 12, 2025: 👋 We've released the inference code and **training code** and weights of MoviiGen1.1.

## 💡 Quickstart

#### Installation
Clone the repo:
```
git clone https://github.com/ZulutionAI/MoviiGen1.1.git
cd MoviiGen1.1
```

Install dependencies:
```
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```

#### Model Download

T2V-14B  Model: 🤗 [Huggingface](https://huggingface.co/ZuluVision/MoviiGen1.1) 
MoviiGen1.1 model supports both 720P and 1080P.

Download models using huggingface-cli:
```
pip install "huggingface_hub[cli]"
huggingface-cli download ZuluVision/MoviiGen1.1 --local-dir ./MoviiGen1.1
```

## 🛠️ Training

### Training Framework

Our training framework is built on [FastVideo](https://github.com/hao-ai-lab/FastVideo), with custom implementation of sequence parallel to optimize memory usage and training efficiency. The sequence parallel approach allows us to distribute the computational load across multiple GPUs, enabling efficient training of large-scale video generation models.

#### Key Features:

- **Sequence Parallel & Ring Attention**: Our custom implementation divides the temporal dimension across multiple GPUs, reducing per-device memory requirements while maintaining model quality.
- **Efficient Data Loading**: Optimized data pipeline for handling high-resolution video frames (Latent Cache and Text Embedding Cache). 
- **Multi Resolution Training Bucket**: Support for training at multiple resolutions.
- **Mixed Precision Training**: Support for BF16/FP16 training to accelerate computation.
- **Distributed Training**: Seamless multi-node, multi-GPU training support.

### Data Preprocessing

We cache the videos and corresponding text prompts as latents and text embeddings to optimize the training process. This preprocessing step significantly improves training efficiency by reducing computational overhead during the training phase.

```bash
cd scripts/data_preprocess
bash scripts/data_preprocess/preprocess.sh
```
Example Data Format:

training_data.json
```json
[
    {
        "cap": "your prompt",
        "path": "path/to/your/video.mp4",
        "resolution": {
            "width": 3840,
            "height": 2160
        },
        "fps": 23.976023976023978,
        "duration": 1.4180833333333331
    },
    ...
]
```
merge.txt:
```txt
relative_path_to_json_dir, training_data.json
```

Output Json:

video_caption.json
```json
[
    {
        "latent_path": "path/to/your/latent.pt",
        "prompt_embed_path": "path/to/your/prompt_embed.pt",
        "length": 12
    },
    ...
]
```

### Train
```bash
bash scripts/train/finetune.sh
```

**When multi-node training, you need to set the number of nodes and the number of processes per node manually.** We provide a sample script for multi-node training.

```bash
bash scripts/bash/finetune_multi_node.sh
```


## Manual Evaluation

<div style="display: flex; justify-content: space-between;">
    <div style="flex: 1; margin-right: 10px;"><img src="assets/movie_asethetic.png" alt="Movie Aesthetic Evaluation" style="width: 100%;" /></div>
    <div style="flex: 1;"><img src="assets/visual_quality.png" alt="Movie Aesthetic Evaluation" style="width: 100%;" /></div>
</div>