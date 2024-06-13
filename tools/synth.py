"""
Helper scripts for generating synthetic images using diffusion model.

Functions:
    - get_top_misclassified
    - get_class_list
    - generateClassPairs
    - outputDirectory
    - pipe_img
    - createPrompts
    - interpolatePrompts
        - slerp
        - get_middle_elements
        - remove_middle
    - genClassImg
    - getMetadata
    - groupbyInterpolation
    - ungroupInterpolation
    - groupAllbyInterpolation
    - getPairIndices
    - generateImagesFromDataset
    - generateTrace
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from DeepCache import DeepCacheSDHelper
from diffusers import (
    LMSDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
)
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchvision import transforms


def get_top_misclassified(val_classifier_json):
    """
    Retrieves the top misclassified classes from a validation classifier JSON file.

    Args:
        val_classifier_json (str): The path to the validation classifier JSON file.

    Returns:
        dict: A dictionary containing the top misclassified classes, where the keys are the class names
              and the values are the number of misclassifications.
    """
    with open(val_classifier_json) as f:
        val_output = json.load(f)
    val_metrics_df = pd.DataFrame.from_dict(
        val_output["val_metrics_details"], orient="index"
    )
    class_dict = dict()
    for k, v in val_metrics_df["top_n_classes"].items():
        class_dict[k] = v
    return class_dict


def get_class_list(val_classifier_json):
    """
    Retrieves the list of classes from the given validation classifier JSON file.

    Args:
        val_classifier_json (str): The path to the validation classifier JSON file.

    Returns:
        list: A sorted list of class names extracted from the JSON file.
    """
    with open(val_classifier_json, "r") as f:
        data = json.load(f)
    return sorted(list(data["val_metrics_details"].keys()))


def generateClassPairs(val_classifier_json):
    """
    Generate pairs of misclassified classes from the given validation classifier JSON.

    Args:
        val_classifier_json (str): The path to the validation classifier JSON file.

    Returns:
        list: A sorted list of pairs of misclassified classes.
    """
    pairs = set()
    misclassified_classes = get_top_misclassified(val_classifier_json)
    for key, value in misclassified_classes.items():
        for v in value:
            pairs.add(tuple(sorted([key, v])))
    return sorted(list(pairs))


def outputDirectory(class_pairs, synth_path, metadata_path):
    """
    Creates the output directory structure for the synthesized data.

    Args:
        class_pairs (list): A list of class pairs.
        synth_path (str): The path to the directory where the synthesized data will be stored.
        metadata_path (str): The path to the directory where the metadata will be stored.

    Returns:
        None
    """
    for id in class_pairs:
        class_folder = f"{synth_path}/{id}"
        if not (os.path.exists(class_folder)):
            os.makedirs(class_folder)
    if not (os.path.exists(metadata_path)):
        os.makedirs(metadata_path)
    print("Info: Output directory ready.")


def pipe_img(
    model_path,
    device="cuda",
    apply_optimization=True,
    use_torchcompile=False,
    ci_cb=(5, 1),
    use_safetensors=None,
    cpu_offload=False,
    scheduler=None,
):
    """
    Creates and returns an image-to-image pipeline for stable diffusion.

    Args:
        model_path (str): The path to the pretrained model.
        device (str, optional): The device to use for computation. Defaults to "cuda".
        apply_optimization (bool, optional): Whether to apply optimization techniques. Defaults to True.
        use_torchcompile (bool, optional): Whether to use torchcompile for model compilation. Defaults to False.
        ci_cb (tuple, optional): A tuple containing the cache interval and cache branch ID. Defaults to (5, 1).
        use_safetensors (bool, optional): Whether to use safetensors. Defaults to None.
        cpu_offload (bool, optional): Whether to enable CPU offloading. Defaults to False.
        scheduler (LMSDiscreteScheduler, optional): The scheduler for the pipeline. Defaults to None.

    Returns:
        StableDiffusionImg2ImgPipeline: The image-to-image pipeline for stable diffusion.
    """
    ###############################
    # Reference:
    # Akimov, R. (2024) Images Interpolation with Stable Diffusion - Hugging Face Open-Source AI Cookbook. Available at: https://huggingface.co/learn/cookbook/en/stable_diffusion_interpolation (Accessed: 4 June 2024).
    ###############################
    if scheduler is None:
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            steps_offset=1,
        )
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        use_safetensors=use_safetensors,
        safety_checker=None,
    ).to(device)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    if apply_optimization:
        # tomesd.apply_patch(pipe, ratio=0.5)
        helper = DeepCacheSDHelper(pipe=pipe)
        cache_interval, cache_branch_id = ci_cb
        helper.set_params(
            cache_interval=cache_interval, cache_branch_id=cache_branch_id
        )  # lower is faster but lower quality
        helper.enable()
        pipe.enable_xformers_memory_efficient_attention()
        if use_torchcompile:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    return pipe


def createPrompts(
    class_name_pairs,
    prompt_structure=None,
    use_default_negative_prompt=False,
    negative_prompt=None,
):
    """
    Create prompts for image generation.

    Args:
        class_name_pairs (list): A list of two class names.
        prompt_structure (str, optional): The structure of the prompt. Defaults to "a photo of a <class_name>".
        use_default_negative_prompt (bool, optional): Whether to use the default negative prompt. Defaults to False.
        negative_prompt (str, optional): The negative prompt to steer the generation away from certain features.

    Returns:
        tuple: A tuple containing two lists - prompts and negative_prompts.
            prompts (list): Text prompts that describe the desired output image.
            negative_prompts (list): Negative prompts that can be used to steer the generation away from certain features.
    """
    if prompt_structure is None:
        prompt_structure = "a photo of a <class_name>"
    elif "<class_name>" not in prompt_structure:
        raise ValueError(
            "The prompt structure must contain the <class_name> placeholder."
        )
    if use_default_negative_prompt:
        default_negative_prompt = (
            "blurry image, disfigured, deformed, distorted, cartoon, drawings"
        )
        negative_prompt = default_negative_prompt

    class1 = class_name_pairs[0]
    class2 = class_name_pairs[1]
    prompt1 = prompt_structure.replace("<class_name>", class1)
    prompt2 = prompt_structure.replace("<class_name>", class2)
    prompts = [prompt1, prompt2]
    if negative_prompt is None:
        print("Info: Negative prompt not provided, returning as None.")
        return prompts, None
    else:
        # Negative prompts that can be used to steer the generation away from certain features.
        negative_prompts = [negative_prompt] * len(prompts)
        return prompts, negative_prompts


def interpolatePrompts(
    prompts,
    pipeline,
    num_interpolation_steps,
    sample_mid_interpolation,
    remove_n_middle=0,
    device="cuda",
):
    """
    Interpolates prompts by generating intermediate embeddings between pairs of prompts.

    Args:
        prompts (List[str]): A list of prompts to be interpolated.
        pipeline: The pipeline object containing the tokenizer and text encoder.
        num_interpolation_steps (int): The number of interpolation steps between each pair of prompts.
        sample_mid_interpolation (int): The number of intermediate embeddings to sample from the middle of the interpolated prompts.
        remove_n_middle (int, optional): The number of middle embeddings to remove from the interpolated prompts. Defaults to 0.
        device (str, optional): The device to run the interpolation on. Defaults to "cuda".

    Returns:
        interpolated_prompt_embeds (torch.Tensor): The interpolated prompt embeddings.
        prompt_metadata (dict): Metadata about the interpolation process, including similarity scores and nearest class information.

    e.g. if num_interpolation_steps = 10, sample_mid_interpolation = 6, remove_n_middle = 2
    Interpolated: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Sampled:            [2, 3, 4, 5, 6, 7]
    Removed:                   x  x
    Returns:            [2, 3,       6, 7]
    """

    ###############################
    # Reference:
    # Akimov, R. (2024) Images Interpolation with Stable Diffusion - Hugging Face Open-Source AI Cookbook. Available at: https://huggingface.co/learn/cookbook/en/stable_diffusion_interpolation (Accessed: 4 June 2024).
    ###############################

    def slerp(v0, v1, num, t0=0, t1=1):
        """
        Performs spherical linear interpolation between two vectors.

        Args:
            v0 (torch.Tensor): The starting vector.
            v1 (torch.Tensor): The ending vector.
            num (int): The number of interpolation points.
            t0 (float, optional): The starting time. Defaults to 0.
            t1 (float, optional): The ending time. Defaults to 1.

        Returns:
            torch.Tensor: The interpolated vectors.

        """
        ###############################
        # Reference:
        # Karpathy, A. (2022) hacky stablediffusion code for generating videos, Gist. Available at: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355 (Accessed: 4 June 2024).
        ###############################
        v0 = v0.detach().cpu().numpy()
        v1 = v1.detach().cpu().numpy()

        def interpolation(t, v0, v1, DOT_THRESHOLD=0.9995):
            """helper function to spherically interpolate two arrays v1 v2"""
            dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
            if np.abs(dot) > DOT_THRESHOLD:
                v2 = (1 - t) * v0 + t * v1
            else:
                theta_0 = np.arccos(dot)
                sin_theta_0 = np.sin(theta_0)
                theta_t = theta_0 * t
                sin_theta_t = np.sin(theta_t)
                s0 = np.sin(theta_0 - theta_t) / sin_theta_0
                s1 = sin_theta_t / sin_theta_0
                v2 = s0 * v0 + s1 * v1
            return v2

        t = np.linspace(t0, t1, num)

        v3 = torch.tensor(np.array([interpolation(t[i], v0, v1) for i in range(num)]))

        return v3

    def get_middle_elements(lst, n):
        """
        Returns a tuple containing a sublist of the middle elements of the given list `lst` and a range of indices of those elements.

        Args:
            lst (list): The list from which to extract the middle elements.
            n (int): The number of middle elements to extract.

        Returns:
            tuple: A tuple containing the sublist of middle elements and a range of indices.

        Raises:
            None

        Examples:
            lst = [1, 2, 3, 4, 5]
            get_middle_elements(lst, 3)
            ([2, 3, 4], range(2, 5))
        """
        if n % 2 == 0:  # Even number of elements
            middle_index = len(lst) // 2 - 1
            start = middle_index - n // 2 + 1
            end = middle_index + n // 2 + 1
            return lst[start:end], range(start, end)
        else:  # Odd number of elements
            middle_index = len(lst) // 2
            start = middle_index - n // 2
            end = middle_index + n // 2 + 1
            return lst[start:end], range(start, end)

    def remove_middle(data, n):
        """
        Remove the middle n elements from a list.

        Args:
            data (list): The input list.
            n (int): The number of elements to remove from the middle of the list.

        Returns:
            list: The modified list with the middle n elements removed.

        Raises:
            ValueError: If n is negative or greater than the length of the list.

        """
        if n < 0 or n > len(data):
            raise ValueError(
                "Invalid value for n. It should be non-negative and less than half the list length"
            )

        # Find the middle index
        middle = len(data) // 2

        # Create slices to exclude the middle n elements
        if n == 1:
            return data[:middle] + data[middle + 1 :]
        elif n % 2 == 0:
            return data[: middle - n // 2] + data[middle + n // 2 :]
        else:
            return data[: middle - n // 2] + data[middle + n // 2 + 1 :]

    batch_size = len(prompts)

    # Tokenizing and encoding prompts into embeddings.
    prompts_tokens = pipeline.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompts_embeds = pipeline.text_encoder(prompts_tokens.input_ids.to(device))[0]

    # Interpolating between embeddings pairs for the given number of interpolation steps.
    interpolated_prompt_embeds = []

    for i in range(batch_size - 1):
        interpolated_prompt_embeds.append(
            slerp(prompts_embeds[i], prompts_embeds[i + 1], num_interpolation_steps)
        )

    full_interpolated_prompt_embeds = interpolated_prompt_embeds[:]
    interpolated_prompt_embeds[0], sample_range = get_middle_elements(
        interpolated_prompt_embeds[0], sample_mid_interpolation
    )

    if remove_n_middle > 0:
        interpolated_prompt_embeds[0] = remove_middle(
            interpolated_prompt_embeds[0], remove_n_middle
        )

    prompt_metadata = dict()
    similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
    for i in range(num_interpolation_steps):
        class1_sim = (
            similarity(
                full_interpolated_prompt_embeds[0][0],
                full_interpolated_prompt_embeds[0][i],
            )
            .mean()
            .item()
        )
        class2_sim = (
            similarity(
                full_interpolated_prompt_embeds[0][num_interpolation_steps - 1],
                full_interpolated_prompt_embeds[0][i],
            )
            .mean()
            .item()
        )
        relative_distance = class1_sim / (class1_sim + class2_sim)

        prompt_metadata[i] = {
            "selected": i in sample_range,
            "similarity": {
                "class1": class1_sim,
                "class2": class2_sim,
                "class1_relative_distance": relative_distance,
                "class2_relative_distance": 1 - relative_distance,
            },
            "nearest_class": int(relative_distance < 0.5),
        }

    interpolated_prompt_embeds = torch.cat(interpolated_prompt_embeds, dim=0).to(device)
    return interpolated_prompt_embeds, prompt_metadata


def genClassImg(
    pipeline,
    pos_embed,
    neg_embed,
    input_image,
    generator,
    latents,
    num_imgs=1,
    height=512,
    width=512,
    num_inference_steps=25,
    guidance_scale=7.5,
):
    """
    Generate class image using the given inputs.

    Args:
        pipeline: The pipeline object used for image generation.
        pos_embed: The positive embedding for the class.
        neg_embed: The negative embedding for the class (optional).
        input_image: The input image for guidance (optional).
        generator: The generator model used for image generation.
        latents: The latent vectors used for image generation.
        num_imgs: The number of images to generate (default is 1).
        height: The height of the generated images (default is 512).
        width: The width of the generated images (default is 512).
        num_inference_steps: The number of inference steps for image generation (default is 25).
        guidance_scale: The scale factor for guidance (default is 7.5).

    Returns:
        The generated class image.
    """

    if neg_embed is not None:
        npe = neg_embed[None, ...]
    else:
        npe = None

    return pipeline(
        height=height,
        width=width,
        num_images_per_prompt=num_imgs,
        prompt_embeds=pos_embed[None, ...],
        negative_prompt_embeds=npe,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        latents=latents,
        image=input_image,
    ).images[0]


def getMetadata(
    class_pairs,
    path,
    seed,
    guidance_scale,
    num_inference_steps,
    num_interpolation_steps,
    sample_mid_interpolation,
    height,
    width,
    prompts,
    negative_prompts,
    pipeline,
    prompt_metadata,
    negative_prompt_metadata,
    ssim_metadata=None,
    save_json=True,
    save_path=".",
):
    """
    Generate metadata for the given parameters.

    Args:
        class_pairs (list): List of class pairs.
        path (str): Path to the data.
        seed (int): Seed value for randomization.
        guidance_scale (float): Scale factor for guidance.
        num_inference_steps (int): Number of inference steps.
        num_interpolation_steps (int): Number of interpolation steps.
        sample_mid_interpolation (bool): Flag to sample mid-interpolation.
        height (int): Height of the image.
        width (int): Width of the image.
        prompts (list): List of prompts.
        negative_prompts (list): List of negative prompts.
        pipeline (object): Pipeline object.
        prompt_metadata (dict): Metadata for prompts.
        negative_prompt_metadata (dict): Metadata for negative prompts.
        ssim_metadata (dict, optional): SSIM scores metadata. Defaults to None.
        save_json (bool, optional): Flag to save metadata as JSON. Defaults to True.
        save_path (str, optional): Path to save the JSON file. Defaults to ".".

    Returns:
        dict: Generated metadata.
    """

    metadata = dict()

    metadata["class_pairs"] = class_pairs
    metadata["path"] = path
    metadata["seed"] = seed
    metadata["params"] = {
        "CFG": guidance_scale,
        "inferenceSteps": num_inference_steps,
        "interpolationSteps": num_interpolation_steps,
        "sampleMidInterpolation": sample_mid_interpolation,
        "height": height,
        "width": width,
    }
    for i in range(len(prompts)):
        metadata[f"prompt_text_{i}"] = prompts[i]
        if negative_prompts is not None:
            metadata[f"negative_prompt_text_{i}"] = negative_prompts[i]
    metadata["pipe_config"] = dict(pipeline.config)
    metadata["prompt_embed_similarity"] = prompt_metadata
    metadata["negative_prompt_embed_similarity"] = negative_prompt_metadata
    if ssim_metadata is not None:
        print("Info: SSIM scores are available.")
        metadata["ssim_scores"] = ssim_metadata
    if save_json:
        with open(
            os.path.join(save_path, f"{'_'.join(i for i in class_pairs)}_{seed}.json"),
            "w",
        ) as f:
            json.dump(metadata, f, indent=4)
    return metadata


def groupbyInterpolation(dir_to_classfolder):
    """
    Group files in a directory by interpolation step.

    Args:
        dir_to_classfolder (str): The path to the directory containing the files.

    Returns:
        None
    """
    files = [
        (f.split(sep="_")[1].split(sep=".")[0], os.path.join(dir_to_classfolder, f))
        for f in os.listdir(dir_to_classfolder)
    ]
    # create a subfolder for each step of the interpolation
    for interpolation_step, file_path in files:
        new_dir = os.path.join(dir_to_classfolder, interpolation_step)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        os.rename(file_path, os.path.join(new_dir, os.path.basename(file_path)))


def ungroupInterpolation(dir_to_classfolder):
    """
    Moves all files from subdirectories within `dir_to_classfolder` to `dir_to_classfolder` itself,
    and then removes the subdirectories.

    Args:
        dir_to_classfolder (str): The path to the directory containing the subdirectories.

    Returns:
        None
    """
    for interpolation_step in os.listdir(dir_to_classfolder):
        if os.path.isdir(os.path.join(dir_to_classfolder, interpolation_step)):
            for f in os.listdir(os.path.join(dir_to_classfolder, interpolation_step)):
                os.rename(
                    os.path.join(dir_to_classfolder, interpolation_step, f),
                    os.path.join(dir_to_classfolder, f),
                )
            os.rmdir(os.path.join(dir_to_classfolder, interpolation_step))


def groupAllbyInterpolation(
    data_path,
    group=True,
    fn_group=groupbyInterpolation,
    fn_ungroup=ungroupInterpolation,
):
    """
    Group or ungroup all data classes by interpolation.

    Args:
        data_path (str): The path to the data.
        group (bool, optional): Whether to group the data. Defaults to True.
        fn_group (function, optional): The function to use for grouping. Defaults to groupbyInterpolation.
        fn_ungroup (function, optional): The function to use for ungrouping. Defaults to ungroupInterpolation.
    """
    data_classes = sorted(os.listdir(data_path))
    if group:
        fn = fn_group
    else:
        fn = fn_ungroup
    for c in data_classes:
        c_path = os.path.join(data_path, c)
        if os.path.isdir(c_path):
            fn(c_path)
            print(f"Processed {c}")


def getPairIndices(subset_len, total_pair_count=1, seed=None):
    """
    Generate pairs of indices for a given subset length.

    Args:
        subset_len (int): The length of the subset.
        total_pair_count (int, optional): The total number of pairs to generate. Defaults to 1.
        seed (int, optional): The seed value for the random number generator. Defaults to None.

    Returns:
        list: A list of pairs of indices.

    """
    rng = np.random.default_rng(seed)
    group_size = (subset_len + total_pair_count - 1) // total_pair_count
    numbers = list(range(subset_len))
    numbers_selection = list(range(subset_len))
    rng.shuffle(numbers)
    for i in range(group_size - subset_len % group_size):
        numbers.append(numbers_selection[i])
    numbers = np.array(numbers)
    groups = numbers[: group_size * total_pair_count].reshape(-1, group_size)
    return groups.tolist()


def generateImagesFromDataset(
    img_subsets,
    class_iterables,
    pipeline,
    interpolated_prompt_embeds,
    interpolated_negative_prompts_embeds,
    num_inference_steps,
    guidance_scale,
    height=512,
    width=512,
    seed=None,
    save_path=".",
    class_pairs=("0", "1"),
    save_image=True,
    image_type="jpg",
    interpolate_range="full",
    device="cuda",
    return_images=False,
):
    """
    Generates images from a dataset using the given parameters.

    Args:
        img_subsets (dict): A dictionary containing image subsets for each class.
        class_iterables (dict): A dictionary containing iterable objects for each class.
        pipeline (object): The pipeline object used for image generation.
        interpolated_prompt_embeds (list): A list of interpolated prompt embeddings.
        interpolated_negative_prompts_embeds (list): A list of interpolated negative prompt embeddings.
        num_inference_steps (int): The number of inference steps for image generation.
        guidance_scale (float): The scale factor for guidance loss during image generation.
        height (int, optional): The height of the generated images. Defaults to 512.
        width (int, optional): The width of the generated images. Defaults to 512.
        seed (int, optional): The seed value for random number generation. Defaults to None.
        save_path (str, optional): The path to save the generated images. Defaults to ".".
        class_pairs (tuple, optional): A tuple containing pairs of class identifiers. Defaults to ("0", "1").
        save_image (bool, optional): Whether to save the generated images. Defaults to True.
        image_type (str, optional): The file format of the saved images. Defaults to "jpg".
        interpolate_range (str, optional): The range of interpolation for prompt embeddings.
            Possible values are "full", "nearest", or "furthest". Defaults to "full".
        device (str, optional): The device to use for image generation. Defaults to "cuda".
        return_images (bool, optional): Whether to return the generated images. Defaults to False.

    Returns:
        dict or tuple: If return_images is True, returns a dictionary containing the generated images for each class and a dictionary containing the SSIM scores for each class and interpolation step.
                       If return_images is False, returns a dictionary containing the SSIM scores for each class and interpolation step.
    """
    if interpolate_range == "nearest":
        nearest_half = True
        furthest_half = False
    elif interpolate_range == "furthest":
        nearest_half = False
        furthest_half = True
    else:
        nearest_half = False
        furthest_half = False

    if seed is None:
        seed = torch.Generator().seed()
    generator = torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Generating initial U-Net latent vectors from a random normal distribution.
    latents = torch.randn(
        (1, pipeline.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(device)

    embed_len = len(interpolated_prompt_embeds)
    embed_pairs = zip(interpolated_prompt_embeds, interpolated_negative_prompts_embeds)
    embed_pairs_list = list(embed_pairs)
    if return_images:
        class_images = dict()
    class_ssim = dict()

    if nearest_half or furthest_half:
        if nearest_half:
            steps_range = (range(0, embed_len // 2), range(embed_len // 2, embed_len))
            mutiplier = 2
        elif furthest_half:
            # uses opposite class of images of the text interpolation
            steps_range = (range(embed_len // 2, embed_len), range(0, embed_len // 2))
            mutiplier = 2
    else:
        steps_range = (range(embed_len), range(embed_len))
        mutiplier = 1

    for class_iter, class_id in enumerate(class_pairs):
        if return_images:
            class_images[class_id] = list()
        class_ssim[class_id] = {
            i: {"ssim_sum": 0, "ssim_count": 0, "ssim_avg": 0} for i in range(embed_len)
        }
        subset_len = len(img_subsets[class_id])
        # to efficiently randomize the steps to interpolate for each image in the class, group_map is used
        # group_map: index is the image id, element is the group id
        # steps_range[class_iter] determines the range of steps to interpolate for the class,
        # so the first half of the steps are for the first class and so on. range(0,7) and range(8,15) for 16 steps
        # then the rest is to multiply the steps to cover the whole subset + remainder
        group_map = (
            list(steps_range[class_iter]) * mutiplier * (subset_len // embed_len + 1)
        )
        rng.shuffle(
            group_map
        )  # shuffle the steps to interpolate for each image, position in the group_map is mapped to the image id

        iter_indices = class_iterables[class_id].pop()
        # generate images for each image in the class, randomly selecting an interpolated step
        for image_id in iter_indices:
            img, trg = img_subsets[class_id][image_id]
            input_image = img.unsqueeze(0)
            interpolate_step = group_map[image_id]
            prompt_embeds, negative_prompt_embeds = embed_pairs_list[interpolate_step]
            generated_image = genClassImg(
                pipeline,
                prompt_embeds,
                negative_prompt_embeds,
                input_image,
                generator,
                latents,
                num_imgs=1,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            pred_image = transforms.ToTensor()(generated_image).unsqueeze(0)
            ssim_score = ssim(pred_image, input_image).item()
            class_ssim[class_id][interpolate_step]["ssim_sum"] += ssim_score
            class_ssim[class_id][interpolate_step]["ssim_count"] += 1
            if return_images:
                class_images[class_id].append(generated_image)
            if save_image:
                if image_type == "jpg":
                    generated_image.save(
                        f"{save_path}/{class_id}/{seed}-{image_id}_{interpolate_step}.{image_type}",
                        format="JPEG",
                        quality=95,
                    )
                elif image_type == "png":
                    generated_image.save(
                        f"{save_path}/{class_id}/{seed}-{image_id}_{interpolate_step}.{image_type}",
                        format="PNG",
                    )
                else:
                    generated_image.save(
                        f"{save_path}/{class_id}/{seed}-{image_id}_{interpolate_step}.{image_type}"
                    )

        # calculate ssim avg for the class
        for i_step in range(embed_len):
            if class_ssim[class_id][i_step]["ssim_count"] > 0:
                class_ssim[class_id][i_step]["ssim_avg"] = (
                    class_ssim[class_id][i_step]["ssim_sum"]
                    / class_ssim[class_id][i_step]["ssim_count"]
                )

    if return_images:
        return class_images, class_ssim
    else:
        return class_ssim


def generateTrace(
    prompts,
    img_subsets,
    class_iterables,
    interpolated_prompt_embeds,
    interpolated_negative_prompts_embeds,
    subset_indices,
    seed=None,
    save_path=".",
    class_pairs=("0", "1"),
    image_type="jpg",
    interpolate_range="full",
    save_prompt_embeds=False,
):
    """
    Generate a trace dictionary containing information about the generated images.

    Args:
        prompts (list): List of prompt texts.
        img_subsets (dict): Dictionary containing image subsets for each class.
        class_iterables (dict): Dictionary containing iterable objects for each class.
        interpolated_prompt_embeds (torch.Tensor): Tensor containing interpolated prompt embeddings.
        interpolated_negative_prompts_embeds (torch.Tensor): Tensor containing interpolated negative prompt embeddings.
        subset_indices (dict): Dictionary containing indices of subsets for each class.
        seed (int, optional): Seed value for random number generation. Defaults to None.
        save_path (str, optional): Path to save the generated images. Defaults to ".".
        class_pairs (tuple, optional): Tuple containing class pairs. Defaults to ("0", "1").
        image_type (str, optional): Type of the generated images. Defaults to "jpg".
        interpolate_range (str, optional): Range of interpolation. Defaults to "full".
        save_prompt_embeds (bool, optional): Flag to save prompt embeddings. Defaults to False.

    Returns:
        dict: Trace dictionary containing information about the generated images.
    """
    trace_dict = {
        "class_pairs": list(),
        "class_id": list(),
        "image_id": list(),
        "interpolation_step": list(),
        "embed_len": list(),
        "pos_prompt_text": list(),
        "neg_prompt_text": list(),
        "input_file_path": list(),
        "output_file_path": list(),
        "input_prompts_embed": list(),
    }

    if interpolate_range == "nearest":
        nearest_half = True
        furthest_half = False
    elif interpolate_range == "furthest":
        nearest_half = False
        furthest_half = True
    else:
        nearest_half = False
        furthest_half = False

    if seed is None:
        seed = torch.Generator().seed()
    rng = np.random.default_rng(seed)

    embed_len = len(interpolated_prompt_embeds)
    embed_pairs = zip(
        interpolated_prompt_embeds.cpu().numpy(),
        interpolated_negative_prompts_embeds.cpu().numpy(),
    )
    embed_pairs_list = list(embed_pairs)

    if nearest_half or furthest_half:
        if nearest_half:
            steps_range = (range(0, embed_len // 2), range(embed_len // 2, embed_len))
            mutiplier = 2
        elif furthest_half:
            # uses opposite class of images of the text interpolation
            steps_range = (range(embed_len // 2, embed_len), range(0, embed_len // 2))
            mutiplier = 2
    else:
        steps_range = (range(embed_len), range(embed_len))
        mutiplier = 1

    for class_iter, class_id in enumerate(class_pairs):

        subset_len = len(img_subsets[class_id])
        # to efficiently randomize the steps to interpolate for each image in the class, group_map is used
        # group_map: index is the image id, element is the group id
        # steps_range[class_iter] determines the range of steps to interpolate for the class,
        # so the first half of the steps are for the first class and so on. range(0,7) and range(8,15) for 16 steps
        # then the rest is to multiply the steps to cover the whole subset + remainder
        group_map = (
            list(steps_range[class_iter]) * mutiplier * (subset_len // embed_len + 1)
        )
        rng.shuffle(
            group_map
        )  # shuffle the steps to interpolate for each image, position in the group_map is mapped to the image id

        iter_indices = class_iterables[class_id].pop()
        # generate images for each image in the class, randomly selecting an interpolated step
        for image_id in iter_indices:
            class_ds = img_subsets[class_id]
            interpolate_step = group_map[image_id]
            sample_count = subset_indices[class_id][0] + image_id
            input_file = os.path.normpath(class_ds.dataset.samples[sample_count][0])
            pos_prompt = prompts[0]
            neg_prompt = prompts[1]
            output_file = f"{save_path}/{class_id}/{seed}-{image_id}_{interpolate_step}.{image_type}"
            if save_prompt_embeds:
                input_prompts_embed = embed_pairs_list[interpolate_step]
            else:
                input_prompts_embed = None

            trace_dict["class_pairs"].append(class_pairs)
            trace_dict["class_id"].append(class_id)
            trace_dict["image_id"].append(image_id)
            trace_dict["interpolation_step"].append(interpolate_step)
            trace_dict["embed_len"].append(embed_len)
            trace_dict["pos_prompt_text"].append(pos_prompt)
            trace_dict["neg_prompt_text"].append(neg_prompt)
            trace_dict["input_file_path"].append(input_file)
            trace_dict["output_file_path"].append(output_file)
            trace_dict["input_prompts_embed"].append(input_prompts_embed)

    return trace_dict
