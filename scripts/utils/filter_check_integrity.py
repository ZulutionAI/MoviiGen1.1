import os.path as osp
import json
import torch
from tqdm import tqdm

prompt_embed_dir = "/cv/data/moviidb_v0.2/preprocess/prompt_with_meta_data_2_5/prompt_embed"

filtered_list = []
video2captions = json.load(open("videos2captions_processed.json", "r"))

for v2c in tqdm(video2captions):

    identifier = v2c["latent_path"]

    if "length" in v2c and v2c["length"] <= 2:
        print(f"skipped too short {v2c['latent_path']}")
        continue

    try:
        metadata_path = osp.join("metadata", identifier[:-2] + "json")
        if not osp.exists(metadata_path):
            print(f"skipped not exist metadata {metadata_path}")
            continue
        metadata = json.load(open(metadata_path, "r"))
    except:
        print(f"skipped error reanding metadata json {identifier[:-2]+'json'}")
        continue

    try:
        latent_path = osp.join("latent", v2c["latent_path"])
        if not osp.exists(latent_path):
            print(f"skipped not exist latent {v2c['latent_path']}")
            continue
        latent = torch.load(latent_path, weights_only=True)
        if latent.shape[1] <= 2:
            print(f"skipped too short {v2c['latent_path']}")
            continue
        if "length" not in v2c:
            v2c["length"] = latent.shape[1]
    except:
        print(f"skipped reading latent error {latent_path}")
        continue

    try:
        prompt_path = osp.join(prompt_embed_dir, identifier)
        if osp.exists(prompt_path):
            prompt = torch.load(prompt_path, weights_only=True)
            v2c["prompt_embed_extended_meta_gemini2.5"] = identifier
        else:
            print(f"skipped not exist prompt embed {identifier}")
    except:
        print(f"skipped reading prompt error {prompt_path}")
        continue

    filtered_list.append(v2c)


json.dump(filtered_list, open("videos2captions_processed_filtered.json", "w"), indent=4)

#  json.dump(json.load(open("final.json", "r")), open("final_wolength.json", "w"), indent=4)
