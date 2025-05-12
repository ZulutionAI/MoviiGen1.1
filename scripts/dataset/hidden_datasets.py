import json
import os
import random

import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):

    def __init__(
        self,
        json_path,
        txt_fixed_len=512
    ):
        self.json_path = json_path
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)
        self.txt_max_len = txt_fixed_len

    def __getitem__(self, idx):
        item = self.data_anno[idx]
        latent = torch.load(item["latent_path"], weights_only=True,map_location='cpu')
        prompt_embed = torch.load(item["prompt_embed_path"], weights_only=True,map_location='cpu')

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

        y = torch.load(item["y_path"], weights_only=True,map_location='cpu')
        clip_feature = torch.load(item["clip_feature_path"], weights_only=True,map_location='cpu')
        
        return latent, prompt_embed, prompt_attention_mask, y, clip_feature

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function(batch):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask
    latents, prompt_embeds, prompt_attention_masks, ys, clip_features = zip(*batch)
    # calculate max shape
    max_t = max([latent.shape[1] for latent in latents])
    max_h = max([latent.shape[2] for latent in latents])
    max_w = max([latent.shape[3] for latent in latents])

    # padding
    latents = [
        torch.nn.functional.pad(
            latent,
            (
                0,
                max_t - latent.shape[1],
                0,
                max_h - latent.shape[2],
                0,
                max_w - latent.shape[3],
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
    ys = torch.stack(ys, dim=0)
    clip_features = torch.stack(clip_features, dim=0)
    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks, ys, clip_features


if __name__ == "__main__":
    dataset = LatentDataset("/cv/zhangpengpeng/cv/video_generation/Wan2.1/data/processed/mixkit/merge_480/videos2caption.json")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=latent_collate_function)
    for latent, prompt_embed, latent_attn_mask, prompt_attention_mask, ys, clip_features in dataloader:
        print(
            latent.shape,
            prompt_embed.shape,
            latent_attn_mask.shape,
            prompt_attention_mask.shape,
            ys.shape,
            clip_features.shape,
        )
        import pdb

        pdb.set_trace()