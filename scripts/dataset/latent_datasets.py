import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):

    def __init__(
        self,
        json_path,
        num_latent_t,
        cfg_rate,
        txt_max_len=512,  # For WanX
        prompt_type="prompt_cap_base_path",
        seed=42,
        resolution_mix=None,
        resolution_mix_p=0.2,
    ):
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)
        self.video_dir = os.path.join(self.datase_dir_path, "video")
        self.latent_dir = os.path.join(self.datase_dir_path, "latent")
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.prompt_attention_mask_dir = os.path.join(self.datase_dir_path, "prompt_attention_mask")

        with open(self.json_path, "r") as f:
            data_annos = json.load(f)

        self.data_anno = []

        if "aspect_ratio_bin" in data_annos[0]:
            keep_ratio = [0, 1]
            aspect_ratios = []
            for anno in data_annos:
                if anno["aspect_ratio_bin"] in keep_ratio:
                    self.data_anno.append(anno)
                    aspect_ratios.append(anno["aspect_ratio_bin"])

            self.aspect_ratios = np.array(aspect_ratios)

        # json.load(f) already keeps the order
        # self.data_anno = sorted(self.data_anno, key=lambda x: x['latent_path'])
        self.num_latent_t = num_latent_t
        self.txt_max_len = txt_max_len
        # just zero embeddings [256, 4096]
        self.uncond_prompt_embed = torch.zeros(512, 4096).to(torch.float32)
        # 512 zeros
        self.uncond_prompt_mask = torch.zeros(512).bool()
        self.lengths = [data_item["length"] if "length" in data_item else 1 for data_item in self.data_anno]
        self.prompt_type = prompt_type

        self.base_seed = seed
        self.resolution_mix = resolution_mix
        self.resolution_mix_p = resolution_mix_p

    # Add this method
    def set_epoch(self, epoch):
        """Sets the current epoch for deterministic random choices."""
        self.epoch = epoch

    def __getitem__(self, idx):
        latent_file = self.data_anno[idx]["latent_path"]

        # prompt_cap_base_path: Glass.2019.2160p.4K.BluRay.x265.10bit.AAC5.1-[YTS.MX]_001059_6611.530_6613.223.pt,
        # prompt_cap_extended_path: "Glass.2019.2160p.4K.BluRay.x265.10bit.AAC5.1-[YTS.MX]_001059_6611.530_6613.223.pt",
        try:
            prompt_embed_file = self.data_anno[idx][self.prompt_type]
            prompt_embed_dir = self.prompt_embed_dir
        except:
            prompt_embed_candidates = self.prompt_type.split(",")
            prompt_embed_file = latent_file
            this_type = np.random.choice(prompt_embed_candidates, 1, p=[0.2, 0.3, 0.5])[0]
            prompt_embed_dir = os.path.join(self.datase_dir_path, this_type)

        if "prompt_attention_mask" in self.data_anno[idx]:
            prompt_attention_mask_file = self.data_anno[idx]["prompt_attention_mask"]
        else:
            prompt_attention_mask_file = None

        latent_dir = os.path.join(self.datase_dir_path, "latent")

        if self.resolution_mix is not None:
            item_epoch_seed = self.base_seed + self.epoch + idx
            local_random = random.Random(item_epoch_seed)
            if local_random.random() < self.resolution_mix_p:
                latent_dir = os.path.join(self.datase_dir_path, self.resolution_mix)

        # load
        latent = torch.load(
            os.path.join(latent_dir, latent_file),
            map_location="cpu",
            weights_only=True,
        )

        latent = latent.squeeze(0)[:, -self.num_latent_t:]
        #  print(f"original latent shape: {original_shape} ==> {latent.shape}")

        if random.random() < self.cfg_rate:
            assert False, "should not enter here"
            prompt_embed = self.uncond_prompt_embed
            prompt_attention_mask = self.uncond_prompt_mask
        else:
            prompt_embed = torch.load(
                os.path.join(prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
            if prompt_attention_mask_file is not None:
                prompt_attention_mask = torch.load(
                    os.path.join(self.prompt_attention_mask_dir, prompt_attention_mask_file),
                    map_location="cpu",
                    weights_only=True,
                )
            else:
                prompt_attention_mask = None

            orig_len = prompt_embed.shape[0]
            if self.txt_max_len > 0:
                embed_dim = prompt_embed.shape[1]

                if orig_len < self.txt_max_len:
                    padding = torch.zeros(self.txt_max_len - orig_len, embed_dim,
                                          device=prompt_embed.device,
                                          dtype=prompt_embed.dtype)
                    prompt_embed = torch.cat([prompt_embed, padding], dim=0)
                elif orig_len > self.txt_max_len:
                    prompt_embed = prompt_embed[:self.txt_max_len]
                    orig_len = self.txt_max_len

                prompt_attention_mask = torch.zeros(self.txt_max_len, dtype=torch.long)
                prompt_attention_mask[:orig_len] = 1
            else:
                prompt_attention_mask = torch.ones(orig_len, dtype=torch.long)
        # print(latent.shape, prompt_embed.shape, prompt_attention_mask.shape)
        return latent, prompt_embed, prompt_attention_mask

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function(batch):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask

    latents, prompt_embeds, prompt_attention_masks = zip(*batch)

    # calculate max shape
    max_t = max([latent.shape[1] for latent in latents])
    max_h = max([latent.shape[2] for latent in latents])
    max_w = max([latent.shape[3] for latent in latents])

    # padding
    latents = [
        torch.nn.functional.pad(
            latent,
            (
                0, max_t - latent.shape[1],
                0, max_h - latent.shape[2],
                0, max_w - latent.shape[3],
            ),
        ) for latent in latents
    ]
    # attn mask
    latent_attn_mask = torch.ones(len(latents), max_t, max_h, max_w)

    # set to 0 if padding
    for i, latent in enumerate(latents):
        latent_attn_mask[i, latent.shape[1]:, :, :] = 0
        latent_attn_mask[i, :, latent.shape[2]:, :] = 0
        latent_attn_mask[i, :, :, latent.shape[3]:] = 0

    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)

    latents = torch.stack(latents, dim=0)

    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks


if __name__ == "__main__":
    dataset = LatentDataset("data/moviidb_v0.1/preprocess/720p/videos2caption.json", num_latent_t=21, cfg_rate=0.0)
    # dataset = LatentDataset("/vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/processed/videos2caption.json", num_latent_t=21, cfg_rate=0.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=latent_collate_function)
    for latent, prompt_embed, latent_attn_mask, prompt_attention_mask in dataloader:
        print(
            latent.shape,
            prompt_embed.shape,
            latent_attn_mask.shape,
            prompt_attention_mask.shape,
        )
        import pdb

        pdb.set_trace()
