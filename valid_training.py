# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import random
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

import wan
from wan.configs import (MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES,
                         WAN_CONFIGS)
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool

warnings.filterwarnings('ignore')

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument("--neg_type", type=str, default="chn")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument("--max_seq_len", type=int, default=75600)
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument("--validation_set", type=str, default="valid_data_t2v")
    parser.add_argument("--base_seed", type=int, default=-1, help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--step_distill",
        action="store_true",
        help="Whether to use step distillation.")
    parser.add_argument(
        "--cfg_distill",
        action="store_true",
        help="Whether to use cfg distillation.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The directory to save the generated image or video to.")
    parser.add_argument(
        "--skip_initial_valid",
        action="store_true",
        help="Whether to skip the initial validation.")
    parser.add_argument(
        "--use_original_model",
        action="store_true",
        help="Whether to use the original model.")
    parser.add_argument(
        "--valid_model_path",
        type=str,
        default=None,
        help="The path to the valid model.")
    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (init_distributed_environment,
                                             initialize_model_parallel)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS[args.task]

    if args.neg_type == "chn":
        cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    elif args.neg_type == "eng":
        cfg.sample_neg_prompt = "Vibrant colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still image, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, merged fingers, motionless frame, cluttered background, three legs, crowded background, walking backwards"
    else:
        raise NotImplementedError

    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    current_iter = 0

    if args.skip_initial_valid:
        for folder in os.listdir(args.output_dir):
            if folder.startswith("checkpoint-"):
                iter_num = int(folder.split('-')[1])
                current_iter = max(current_iter, iter_num)

    while True:

        # 找到output_dir下的最大checkpoint文件夹
        max_iter = current_iter
        max_checkpoint_path = None

        if os.path.exists(os.path.join(args.output_dir, 'diffusion_pytorch_model.safetensors')):
            max_checkpoint_path = os.path.join(args.output_dir, 'diffusion_pytorch_model.safetensors')
            max_iter = int(args.output_dir.split("_")[1])
        else:

            for folder in os.listdir(args.output_dir):
                if folder.startswith("checkpoint-step_"):
                    iter_num = int(folder.split('_')[1])
                    if iter_num > max_iter:
                        max_iter = iter_num
                        max_checkpoint_path = os.path.join(
                            args.output_dir, folder, 'diffusion_pytorch_model.safetensors')
                elif folder.startswith("checkpoint-"):
                    iter_num = int(folder.split('-')[1])
                    if iter_num > max_iter:
                        max_iter = iter_num
                        max_checkpoint_path = os.path.join(
                            args.output_dir, folder, 'diffusion_pytorch_model.safetensors')

        # 如果找到新的最大checkpoint，进行验证
        if max_checkpoint_path and max_iter > current_iter:
            if world_size > 1:
                dist.barrier()

            if world_size > 1:
                max_checkpoint_path = [max_checkpoint_path]
                dist.broadcast_object_list(max_checkpoint_path, src=0)
                max_checkpoint_path = max_checkpoint_path[0]

            logging.info("Creating WanT2V pipeline.")
            if args.use_original_model:
                weight_path = None
            elif args.valid_model_path:
                weight_path = args.valid_model_path
            else:
                weight_path = max_checkpoint_path

            wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
                weight_path=weight_path,
                STEP_DISTILL=args.step_distill,
                CFG_DISTILL=args.cfg_distill,
            )

            logging.info(
                f"Generating {'image' if 't2i' in args.task else 'video'} ...")

            prompts = []
            prompts_filenames = []
            prompt_files = natsorted(os.listdir(f"assets/{args.validation_set}"))

            for file in prompt_files:
                if file.endswith('.txt'):
                    prompt_file = os.path.join(f"assets/{args.validation_set}", file)
                    if os.path.exists(prompt_file):
                        with open(prompt_file, 'r') as f:
                            lines = [ln.strip() for ln in f.readlines()]
                            prompt = " ".join(lines)
                            prompts.append(prompt)
                            prompts_filenames.append(file.split('.')[0])

            dist.broadcast_object_list(prompts, src=0)
            dist.broadcast_object_list(prompts_filenames, src=0)

            if rank == 0:
                pbar = tqdm(prompts)
            else:
                pbar = prompts

            if not args.base_seed >= 0:
                random_seed = True
            else:
                random_seed = False

            for i, prompt in enumerate(pbar):

                if dist.is_initialized():

                    if random_seed:
                        seed_bytes = os.urandom(8)
                        args.base_seed = int.from_bytes(seed_bytes, sys.byteorder) % (2**32 - 1)

                    base_seed = [args.base_seed] if rank == 0 else [None]
                    dist.broadcast_object_list(base_seed, src=0)
                    args.base_seed = base_seed[0]

                print("set seed to :", args.base_seed, "\n")

                video = wan_t2v.generate(
                    prompt,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model,
                    seq_len=args.max_seq_len,
                )
                if not isinstance(pbar, list):
                    pbar.update(1)
                dist.barrier()

                if rank == 0:

                    valid_folder = os.path.join(
                        '/'.join(max_checkpoint_path.split('/')[:-1]),
                        args.validation_set,
                    )
                    os.makedirs(valid_folder, exist_ok=True)

                    suffix = 'png' if "t2i" in args.task else 'mp4'

                    if args.use_original_model:
                        save_file = os.path.join(
                            valid_folder,
                            f'{prompts_filenames[i]}_{args.size}_original_seed{args.base_seed}.{suffix}'
                        )
                    else:
                        save_file = os.path.join(
                            valid_folder,
                            f'{prompts_filenames[i]}_{args.size}_seed{args.base_seed}.{suffix}',
                        )

                    if "t2i" in args.task:
                        logging.info(f"Saving generated image to {save_file}")
                        cache_image(
                            tensor=video.squeeze(1)[None],
                            save_file=save_file,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1),
                        )
                    else:
                        logging.info(f"Saving generated video to {save_file}")
                        cache_video(
                            tensor=video[None],
                            save_file=save_file,
                            fps=cfg.sample_fps,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1)
                        )

                dist.barrier()

            if world_size > 1:
                dist.barrier()

            current_iter = max_iter
            del wan_t2v
            torch.cuda.empty_cache()

        time.sleep(10)


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
