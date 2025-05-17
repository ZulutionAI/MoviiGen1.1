# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional, Union

import dashscope
import torch
from PIL import Image

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_VER = 2
except ModuleNotFoundError:
    flash_attn_varlen_func = None  # in compatible with CPU machines
    FLASH_VER = None

LM_EN_SYS_PROMPT = "You are an advanced AI model tasked with generating and extending structured and detailed video captions. You must respond in the language used by the user."

@dataclass
class PromptOutput(object):
    status: bool
    prompt: str
    seed: int
    system_prompt: str
    message: str

    def add_custom_field(self, key: str, value) -> None:
        self.__setattr__(key, value)


class PromptExpander:

    def __init__(self, model_name, is_vl=False, device=0, **kwargs):
        self.model_name = model_name
        self.is_vl = is_vl
        self.device = device

    def extend_with_img(self,
                        prompt,
                        system_prompt,
                        image=None,
                        seed=-1,
                        *args,
                        **kwargs):
        pass

    def extend(self, prompt, system_prompt, seed=-1, *args, **kwargs):
        pass

    def decide_system_prompt(self, tar_lang="en"):
        return LM_EN_SYS_PROMPT

    def __call__(self,
                 prompt,
                 tar_lang="en",
                 image=None,
                 seed=-1,
                 *args,
                 **kwargs):
        system_prompt = self.decide_system_prompt(tar_lang=tar_lang)
        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        if image is not None and self.is_vl:
            return self.extend_with_img(
                prompt, system_prompt, image=image, seed=seed, *args, **kwargs)
        elif not self.is_vl:
            return self.extend(prompt, system_prompt, seed, *args, **kwargs)
        else:
            raise NotImplementedError


class QwenPromptExpander(PromptExpander):

    def __init__(self, model_name=None, device=0, is_vl=False, **kwargs):
        '''
        Args:
            model_name: Use predefined model names such as 'QwenVL2.5_7B' and 'Qwen2.5_14B',
                which are specific versions of the Qwen model. Alternatively, you can use the
                local path to a downloaded model or the model name from Hugging Face."
              Detailed Breakdown:
                Predefined Model Names:
                * 'QwenVL2.5_7B' and 'Qwen2.5_14B' are specific versions of the Qwen model.
                Local Path:
                * You can provide the path to a model that you have downloaded locally.
                Hugging Face Model Name:
                * You can also specify the model name from Hugging Face's model hub.
            is_vl: A flag indicating whether the task involves visual-language processing.
            **kwargs: Additional keyword arguments that can be passed to the function or method.
        '''
        if model_name is None:
            model_name = 'ZuluVision/MoviiGen1.1_Prompt_Rewriter'
        super().__init__(model_name, is_vl, device, **kwargs)
        self.model_name = model_name

        if self.is_vl:
            raise NotImplementedError("VL is not supported")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
            if "AWQ" in self.model_name else "auto",
            attn_implementation="flash_attention_2"
            if FLASH_VER == 2 else None,
            device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def extend(self, prompt, system_prompt, seed=-1, *args, **kwargs):
        self.model = self.model.to(self.device)
        messages = [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": prompt
        }]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text],
                                      return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids)
        ]

        expanded_prompt = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        self.model = self.model.to("cpu")
        return PromptOutput(
            status=True,
            prompt=expanded_prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=json.dumps({"content": expanded_prompt},
                               ensure_ascii=False))

    def extend_with_img(self,
                        prompt,
                        system_prompt,
                        image: Union[Image.Image, str] = None,
                        seed=-1,
                        *args,
                        **kwargs):
        self.model = self.model.to(self.device)
        messages = [{
            'role': 'system',
            'content': [{
                "type": "text",
                "text": system_prompt
            }]
        }, {
            "role":
                "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        expanded_prompt = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        self.model = self.model.to("cpu")
        return PromptOutput(
            status=True,
            prompt=expanded_prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=json.dumps({"content": expanded_prompt},
                               ensure_ascii=False))


if __name__ == "__main__":

    seed = 100
    prompt = "夏日海滩度假风格，一只戴着墨镜的白色猫咪坐在冲浪板上。猫咪毛发蓬松，表情悠闲，直视镜头。背景是模糊的海滩景色，海水清澈，远处有绿色的山丘和蓝天白云。猫咪的姿态自然放松，仿佛在享受海风和阳光。近景特写，强调猫咪的细节和海滩的清新氛围。"
    en_prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    # test cases for prompt extend
    ds_model_name = "qwen-plus"
    # for qwenmodel, you can download the model form modelscope or huggingface and use the model path as model_name
    qwen_model_name = "./models/Qwen2.5-14B-Instruct/"  # VRAM: 29136MiB
    # qwen_model_name = "./models/Qwen2.5-14B-Instruct-AWQ/"  # VRAM: 10414MiB

    # test dashscope api
    dashscope_prompt_expander = DashScopePromptExpander(
        model_name=ds_model_name)
    dashscope_result = dashscope_prompt_expander(prompt, tar_lang="ch")
    print("LM dashscope result -> ch",
          dashscope_result.prompt)  #dashscope_result.system_prompt)
    dashscope_result = dashscope_prompt_expander(prompt, tar_lang="en")
    print("LM dashscope result -> en",
          dashscope_result.prompt)  #dashscope_result.system_prompt)
    dashscope_result = dashscope_prompt_expander(en_prompt, tar_lang="ch")
    print("LM dashscope en result -> ch",
          dashscope_result.prompt)  #dashscope_result.system_prompt)
    dashscope_result = dashscope_prompt_expander(en_prompt, tar_lang="en")
    print("LM dashscope en result -> en",
          dashscope_result.prompt)  #dashscope_result.system_prompt)
    # # test qwen api
    qwen_prompt_expander = QwenPromptExpander(
        model_name=qwen_model_name, is_vl=False, device=0)
    qwen_result = qwen_prompt_expander(prompt, tar_lang="ch")
    print("LM qwen result -> ch",
          qwen_result.prompt)  #qwen_result.system_prompt)
    qwen_result = qwen_prompt_expander(prompt, tar_lang="en")
    print("LM qwen result -> en",
          qwen_result.prompt)  # qwen_result.system_prompt)
    qwen_result = qwen_prompt_expander(en_prompt, tar_lang="ch")
    print("LM qwen en result -> ch",
          qwen_result.prompt)  #, qwen_result.system_prompt)
    qwen_result = qwen_prompt_expander(en_prompt, tar_lang="en")
    print("LM qwen en result -> en",
          qwen_result.prompt)  # , qwen_result.system_prompt)
    # test case for prompt-image extend
    ds_model_name = "qwen-vl-max"
    #qwen_model_name = "./models/Qwen2.5-VL-3B-Instruct/" #VRAM: 9686MiB
    qwen_model_name = "./models/Qwen2.5-VL-7B-Instruct-AWQ/"  # VRAM: 8492
    image = "./examples/i2v_input.JPG"

    # test dashscope api why image_path is local directory; skip
    dashscope_prompt_expander = DashScopePromptExpander(
        model_name=ds_model_name, is_vl=True)
    dashscope_result = dashscope_prompt_expander(
        prompt, tar_lang="ch", image=image, seed=seed)
    print("VL dashscope result -> ch",
          dashscope_result.prompt)  #, dashscope_result.system_prompt)
    dashscope_result = dashscope_prompt_expander(
        prompt, tar_lang="en", image=image, seed=seed)
    print("VL dashscope result -> en",
          dashscope_result.prompt)  # , dashscope_result.system_prompt)
    dashscope_result = dashscope_prompt_expander(
        en_prompt, tar_lang="ch", image=image, seed=seed)
    print("VL dashscope en result -> ch",
          dashscope_result.prompt)  #, dashscope_result.system_prompt)
    dashscope_result = dashscope_prompt_expander(
        en_prompt, tar_lang="en", image=image, seed=seed)
    print("VL dashscope en result -> en",
          dashscope_result.prompt)  # , dashscope_result.system_prompt)
    # test qwen api
    qwen_prompt_expander = QwenPromptExpander(
        model_name=qwen_model_name, is_vl=True, device=0)
    qwen_result = qwen_prompt_expander(
        prompt, tar_lang="ch", image=image, seed=seed)
    print("VL qwen result -> ch",
          qwen_result.prompt)  #, qwen_result.system_prompt)
    qwen_result = qwen_prompt_expander(
        prompt, tar_lang="en", image=image, seed=seed)
    print("VL qwen result ->en",
          qwen_result.prompt)  # , qwen_result.system_prompt)
    qwen_result = qwen_prompt_expander(
        en_prompt, tar_lang="ch", image=image, seed=seed)
    print("VL qwen vl en result -> ch",
          qwen_result.prompt)  #, qwen_result.system_prompt)
    qwen_result = qwen_prompt_expander(
        en_prompt, tar_lang="en", image=image, seed=seed)
    print("VL qwen vl en result -> en",
          qwen_result.prompt)  # , qwen_result.system_prompt)
