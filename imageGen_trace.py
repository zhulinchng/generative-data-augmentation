"""
For debugging and analysis purposes, this script generates the trace for the image generation process.
Refer to imageGen.py for the main image generation script.
"""

from collections import Counter

import pandas as pd
import torch
from tqdm import tqdm

from tools import classes, data, synth

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

### Load Data ###

dataset_type = "data/imagewoof"  # path to dataset

save_prompt_embeds = False

cache_path = f"./{dataset_type}"
genInput_dir = f"{dataset_type}/train"
synth_path = f"{dataset_type}/synthetic"
metadata_path = f"{dataset_type}/metadata"
val_classifier_json = f"{dataset_type}/val.json"
# get the top 5 misclassified classes from dev from base classifier
class_list = synth.get_class_list(val_classifier_json)
# combine all misclassified class pairs
class_pairs_combo = synth.generateClassPairs(val_classifier_json)
data.cacheGenData(
    genInput_dir, "imagenet_inputImg", save_path=cache_path, resize=(512, 512)
)
genInput_dataset = data.loadData("imagenet_inputImg", cache_path=cache_path)

img_subsets = data.getSubsets(genInput_dataset, genInput_dir)
subset_indices = data.getSubsetIndicies(genInput_dataset, genInput_dir)
print(f"Running for {dataset_type}.")

### Parameters for Image Generation ###

# prompt_format = "a photo of a <class_name>"  # Learning Transferable Visual Models From Natural Language Supervision
prompt_format = "a photo of a <class_name>, a type of dog"  # Learning Transferable Visual Models From Natural Language Supervision

negative_prompt = "blurry image, disfigured, deformed, distorted, cartoon, drawings"  # adapted from 10.1007/978-3-031-44237-7_14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id_or_path = "runwayml/stable-diffusion-v1-5"
# model_id_or_path = "./models/stable-diffusion-v1-5"  # if available locally

height, width = 512, 512  # set the optimal image size for the model

# The guidance scale is set to its normal range (7 - 10).
guidance_scale = 8

# The number of inference steps was chosen empirically to generate an acceptable picture within an acceptable time.
num_inference_steps = 25

# Again, these values were chosen empirically.
num_interpolation_steps = 16
sample_mid_interpolation = 16
remove_n_middle = 0

### Interpolation Checks ###

assert num_interpolation_steps % 2 == 0
assert sample_mid_interpolation % 2 == 0
assert remove_n_middle % 2 == 0
assert num_interpolation_steps >= sample_mid_interpolation
assert num_interpolation_steps >= 2 and sample_mid_interpolation >= 2
assert sample_mid_interpolation - remove_n_middle >= 2

# Setup pipeline

pipe = synth.pipe_img(model_id_or_path, device=device)

# To reproduce: 4796730343513556238 for woof, 1127962904372660145 for stanford dogs, 18316237598377439927 for imagenette
seed = torch.Generator().seed()
print(f"Seed: {seed}")

class_iterables = dict()
for c in class_list:
    total_pair_count = Counter(c == x or c == y for x, y in class_pairs_combo)[True]
    class_iterables[c] = synth.getPairIndices(
        len(img_subsets[c]), total_pair_count, seed=seed
    )

df = None

# Generate Images for each class pair
for combo_iter, class_pairs in enumerate(tqdm(class_pairs_combo)):
    class_name_pairs = (
        # IMAGENET2012_CLASSES, STANFORD_DOGS
        classes.IMAGENET2012_CLASSES[class_pairs[0]],
        classes.IMAGENET2012_CLASSES[class_pairs[1]],
    )
    synth.outputDirectory(class_pairs, synth_path, metadata_path)
    prompts, negative_prompts = synth.createPrompts(
        class_name_pairs,
        prompt_structure=prompt_format,
        negative_prompt=negative_prompt,
    )
    print(f"Generating images for {prompts[0]} and {prompts[1]}.")
    interpolated_prompt_embeds, prompt_metadata = synth.interpolatePrompts(
        prompts,
        pipe,
        num_interpolation_steps,
        sample_mid_interpolation,
        remove_n_middle=remove_n_middle,
        device=device,
    )
    if negative_prompts is not None:
        interpolated_negative_prompts_embeds, negative_prompt_metadata = (
            synth.interpolatePrompts(
                negative_prompts,
                pipe,
                num_interpolation_steps,
                sample_mid_interpolation,
                remove_n_middle=remove_n_middle,
                device=device,
            )
        )
    else:
        interpolated_negative_prompts_embeds, negative_prompt_metadata = [None] * len(
            interpolated_prompt_embeds
        ), None

    trace_dict = synth.generateTrace(
        prompts,
        img_subsets,
        class_iterables,
        interpolated_prompt_embeds,
        interpolated_negative_prompts_embeds,
        subset_indices,
        seed=seed,
        save_path=synth_path,
        class_pairs=class_pairs,
        image_type="jpg",
        interpolate_range="nearest",
        save_prompt_embeds=save_prompt_embeds,
    )
    if df is None:
        df = pd.DataFrame.from_dict(trace_dict)
    else:
        df = pd.concat([df, pd.DataFrame.from_dict(trace_dict)], ignore_index=True)

df = df.drop_duplicates(
    subset=["output_file_path"], keep="last"
)  # keep the last row as the image was overwritten by the last iteration
if save_prompt_embeds:
    df.to_pickle(f"{metadata_path}/imageGen_trace.pkl")
else:
    df.to_csv(f"{metadata_path}/imageGen_trace.csv", index=False)
