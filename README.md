
# MoviiGen 1.1

[**MoviiGen 1.1: Towards Cinematic-Quality Video Generative Models**]("https://huggingface.co/ZuluVision/MoviiGen1.1") <be>

In this repository, we present **MoviiGen 1.1**, a cutting-edge video generation model that excels in cinematic aesthetics and visual quality. This model is a fine-tuning model based on the Wan2.1. Based on comprehensive evaluations by 11 professional filmmakers and AIGC creators, including industry experts, across 60 aesthetic dimensions, **MoviiGen 1.1** demonstrates superior performance in key cinematic aspects:

- 👍 **Superior Cinematic Aesthetics**: **MoviiGen 1.1** outperforms competitors in three critical dimensions: atmosphere creation, camera movement, and object detail preservation, making it the preferred choice for professional cinematic applications. 
- 👍 **Visual Coherence & Quality**: MoviiGen 1.1 excels in clarity (+14.6%) and realism (+4.3%), making it ideal for high-fidelity scenarios such as real-scene conversion and portrait detail. Wan2.1 stands out in smoothness and overall visual harmony, better suited for tasks emphasizing composition, coherence, and artistic style. Both models have close overall scores, so users can select MoviiGen 1.1 for clarity and realism, or Wan2.1 for style and structural consistency.
- 👍 **Comprehensive Visual Capabilities**: **MoviiGen 1.1** provides stable performance in complex visual scenarios, ensuring consistent subject and scene representation while maintaining high-quality motion dynamics.
- 👍 **High-Quality Output**: The model generates videos with exceptional clarity and detail, supporting both 720P and 1080P resolutions while maintaining consistent visual quality throughout the sequence.
- 👍 **Professional-Grade Results**: **MoviiGen 1.1** is particularly well-suited for applications where cinematic quality, visual coherence, and aesthetic excellence are paramount, offering superior overall quality compared to other models.

This repository features our latest model, which establishes new benchmarks in cinematic video generation. Through extensive evaluation by industry professionals, it has demonstrated exceptional capabilities in creating high-quality visuals with natural motion dynamics and consistent aesthetic quality, making it an ideal choice for professional video production and creative applications.

## Video Demos

| <video width="320" controls><source src="assets/79_1920*1056_seed3732225395.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="assets/150_1920*1056_seed1674457713.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="assets/143_1920*1056_seed3114534932.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="assets/94_1920*1056_seed3693446494.mp4" type="video/mp4">Your browser does not support the video tag.</video> |
|--------|--------|--------|--------|
| <video width="320" controls><source src="assets/23_1920*1056_seed3934691816.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="assets/13_1920*1056..mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="assets/26_1920*1056..mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="assets/39_1920*1056..mp4" type="video/mp4">Your browser does not support the video tag.</video> |
|--------|--------|--------|--------|
| <video width="320" controls><source src="assets/100_1920*1056_seed2949593166.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="assets/54_1920*1056..mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="assets/107_1920*1056_seed525896597.mp4" type="video/mp4">Your browser does not support the video tag.</video> | <video width="320" controls><source src="assets/94_1920*1056_seed3693446494.mp4" type="video/mp4">Your browser does not support the video tag.</video> |
|--------|--------|--------|--------|

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

T2V-14B  Model: 🤗 [Huggingface](https://huggingface.co/ZuluVision/MoviiGen1.1)          | Supports both 720P and 1080P


Download models using huggingface-cli:
```
pip install "huggingface_hub[cli]"
huggingface-cli download ZuluVision/MoviiGen1.1 --local-dir ./MoviiGen1.1
```

#### Run Text-to-Video Generation

This repository supports two Text-to-Video models (14B) and two resolutions (720P and 1080P). The parameters and configurations for these models are as follows:

<table>
    <thead>
        <tr>
            <th rowspan="2">Task</th>
            <th colspan="2">Resolution</th>
            <th rowspan="2">Model</th>
        </tr>
        <tr>
            <th>720P</th>
            <th>1080P</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>t2v-14B</td>
            <td style="color: green;">✔️</td>
            <td style="color: green;">✔️</td>
            <td>Wan2.1-T2V-14B</td>
        </tr>
    </tbody>
</table>


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

Example

### Training




## Manual Evaluation

<div style="display: flex; justify-content: space-between;">
    <div style="flex: 1; margin-right: 10px;"><img src="assets/movie_asethetic.png" alt="Movie Aesthetic Evaluation" style="width: 100%;" /></div>
    <div style="flex: 1;"><img src="assets/visual_quality.png" alt="Movie Aesthetic Evaluation" style="width: 100%;" /></div>
</div>



## Community Contributions
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) provides more support for Wan, including video-to-video, FP8 quantization, VRAM optimization, LoRA training, and more. Please refer to [their examples](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo).



