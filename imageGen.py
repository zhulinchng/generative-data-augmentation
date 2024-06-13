"""
Script to generate synthetic images for the specified dataset.

The script generates synthetic images for each class pair of the confusing/top-misclassified classes in the dataset.
Each run of the script generates approximately 1:1 ratio of train-to-synthetic images.

While the script works in both Windows and Linux, it is recommended to run the script in Linux for better performance as torch.compile() is not supported in Windows.
"""

from collections import Counter

import torch
from tqdm import tqdm

from tools import classes, data, synth

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

### Load Data ###

dataset_type = "data/imagewoof"  # path to dataset

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

print(f"Running for {dataset_type}.")

### Parameters for Image Generation ###

# prompt adapted from Learning Transferable Visual Models From Natural Language Supervision
# prompt_format = "a photo of a <class_name>"
prompt_format = "a photo of a <class_name>, a type of dog"
# add ", a type of dog" for imagewoof/stanford dogs

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

# Generate Images for each class pair
for combo_iter, class_pairs in enumerate(tqdm(class_pairs_combo)):
    class_name_pairs = (
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

    ssim_scores = synth.generateImagesFromDataset(
        img_subsets,
        class_iterables,
        pipe,
        interpolated_prompt_embeds,
        interpolated_negative_prompts_embeds,
        num_inference_steps,
        guidance_scale,
        height=height,
        width=width,
        seed=seed,
        save_path=synth_path,
        class_pairs=class_pairs,
        save_image=True,
        image_type="jpg",
        interpolate_range="nearest",
        device=device,
        return_images=False,
    )

    metadata = synth.getMetadata(
        class_pairs,
        synth_path,
        seed,
        guidance_scale,
        num_inference_steps,
        num_interpolation_steps,
        sample_mid_interpolation,
        height,
        width,
        prompts,
        negative_prompts,
        pipe,
        prompt_metadata,
        negative_prompt_metadata,
        ssim_scores,
        save_json=True,
        save_path=metadata_path,
    )
