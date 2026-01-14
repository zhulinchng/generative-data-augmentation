"""
Generate synthetic images for dataset augmentation using Stable Diffusion.

This script generates synthetic images for confusing class pairs identified from
validation metrics. It uses image-to-image diffusion with prompt interpolation
to create training data that helps improve classifier performance.

Note: Linux is recommended for optimal performance (torch.compile support).
"""

from collections import Counter

import torch
from tqdm import tqdm

from tools import classes, data, synth

# Enable performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Configuration
DATASET_TYPE = "data/imagewoof"  # Path to dataset

# Setup paths
cache_path = f"./{DATASET_TYPE}"
genInput_dir = f"{DATASET_TYPE}/train"
synth_path = f"{DATASET_TYPE}/synthetic"
metadata_path = f"{DATASET_TYPE}/metadata"
val_classifier_json = f"{DATASET_TYPE}/val.json"

# Load class pairs from validation metrics
class_list = synth.get_class_list(val_classifier_json)
class_pairs_combo = synth.generateClassPairs(val_classifier_json)

# Cache and load input dataset
data.cacheGenData(
    genInput_dir, "imagenet_inputImg", save_path=cache_path, resize=(512, 512)
)
genInput_dataset = data.loadData("imagenet_inputImg", cache_path=cache_path)
img_subsets = data.getSubsets(genInput_dataset, genInput_dir)

print(f"Running for {DATASET_TYPE}.")

# Image Generation Parameters
# Prompt format adapted from "Learning Transferable Visual Models From Natural Language Supervision"
PROMPT_FORMAT = (
    "a photo of a <class_name>, a type of dog"  # Add ", a type of dog" for dog datasets
)
NEGATIVE_PROMPT = "blurry image, disfigured, deformed, distorted, cartoon, drawings"

# Model configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # Or use "./models/stable-diffusion-v1-5" for local

# Generation parameters
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
GUIDANCE_SCALE = 8  # Guidance scale in normal range (7-10)
NUM_INFERENCE_STEPS = 25  # Empirically chosen for quality/speed balance

# Interpolation parameters
NUM_INTERPOLATION_STEPS = 16
SAMPLE_MID_INTERPOLATION = 16
REMOVE_N_MIDDLE = 0

# Validate interpolation parameters
assert NUM_INTERPOLATION_STEPS % 2 == 0, "Interpolation steps must be even"
assert SAMPLE_MID_INTERPOLATION % 2 == 0, "Sample mid interpolation must be even"
assert REMOVE_N_MIDDLE % 2 == 0, "Remove n middle must be even"
assert (
    NUM_INTERPOLATION_STEPS >= SAMPLE_MID_INTERPOLATION
), "Interpolation steps must be >= sample mid"
assert (
    NUM_INTERPOLATION_STEPS >= 2 and SAMPLE_MID_INTERPOLATION >= 2
), "Minimum 2 steps required"
assert SAMPLE_MID_INTERPOLATION - REMOVE_N_MIDDLE >= 2, "Must keep at least 2 samples"

# Initialize pipeline
pipe = synth.pipe_img(MODEL_ID, device=DEVICE)

# Set random seed for reproducibility
# Seeds: 4796730343513556238 (woof), 1127962904372660145 (stanford dogs), 18316237598377439927 (imagenette)
seed = torch.Generator().seed()
print(f"Seed: {seed}")

# Prepare class iterables for pair generation
class_iterables = {}
for class_id in class_list:
    total_pair_count = Counter(
        class_id == x or class_id == y for x, y in class_pairs_combo
    )[True]
    class_iterables[class_id] = synth.getPairIndices(
        len(img_subsets[class_id]), total_pair_count, seed=seed
    )

# Generate images for each class pair
for combo_iter, class_pairs in enumerate(tqdm(class_pairs_combo)):
    # Get class names from ImageNet mapping
    class_name_pairs = (
        classes.IMAGENET2012_CLASSES[class_pairs[0]],
        classes.IMAGENET2012_CLASSES[class_pairs[1]],
    )

    # Setup output directories
    synth.outputDirectory(class_pairs, synth_path, metadata_path)

    # Create prompts for the class pair
    prompts, negative_prompts = synth.createPrompts(
        class_name_pairs,
        prompt_structure=PROMPT_FORMAT,
        negative_prompt=NEGATIVE_PROMPT,
    )
    print(f"Generating images for {prompts[0]} and {prompts[1]}.")

    # Interpolate positive prompts
    interpolated_prompt_embeds, prompt_metadata = synth.interpolatePrompts(
        prompts,
        pipe,
        NUM_INTERPOLATION_STEPS,
        SAMPLE_MID_INTERPOLATION,
        remove_n_middle=REMOVE_N_MIDDLE,
        device=DEVICE,
    )

    # Interpolate negative prompts if provided
    if negative_prompts is not None:
        interpolated_negative_prompts_embeds, negative_prompt_metadata = (
            synth.interpolatePrompts(
                negative_prompts,
                pipe,
                NUM_INTERPOLATION_STEPS,
                SAMPLE_MID_INTERPOLATION,
                remove_n_middle=REMOVE_N_MIDDLE,
                device=DEVICE,
            )
        )
    else:
        interpolated_negative_prompts_embeds = [None] * len(interpolated_prompt_embeds)
        negative_prompt_metadata = None

    # Generate synthetic images
    ssim_scores = synth.generateImagesFromDataset(
        img_subsets,
        class_iterables,
        pipe,
        interpolated_prompt_embeds,
        interpolated_negative_prompts_embeds,
        NUM_INFERENCE_STEPS,
        GUIDANCE_SCALE,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        seed=seed,
        save_path=synth_path,
        class_pairs=class_pairs,
        save_image=True,
        image_type="jpg",
        interpolate_range="nearest",
        device=DEVICE,
        return_images=False,
    )

    # Save metadata for this class pair
    metadata = synth.getMetadata(
        class_pairs,
        synth_path,
        seed,
        GUIDANCE_SCALE,
        NUM_INFERENCE_STEPS,
        NUM_INTERPOLATION_STEPS,
        SAMPLE_MID_INTERPOLATION,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        prompts,
        negative_prompts,
        pipe,
        prompt_metadata,
        negative_prompt_metadata,
        ssim_scores,
        save_json=True,
        save_path=metadata_path,
    )
