import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import sys
import json
import os
import logging

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        print("Usage: python eval_internvl.py test_file.json data_folder")
        sys.exit(1)

    test_file = args[1] # this sets which subset to evaluate
    data_folder = args[2] # this sets full frame or cropped image
    print(f'test_file: {test_file}')
    print(f'data_folder: {data_folder}')

    # load the test data
    with open(test_file) as f:
        test_data = json.load(f)['eval_cases']

    save_filename = ('result_' + test_file.split('/')[-1]).replace('json', 'csv')
    print(f'save_filename: {save_filename}')

    # load prompt
    with open('prompt.txt') as f:
        prompt_template = f.read()

    # load model
    path = "OpenGVLab/InternVL3_5-8B"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    print('Model loaded successfully!')

    # question = 'Hi how are you?'
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    # response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    # print(f'User: {question}\nAssistant: {response}')

    with open(save_filename, 'w') as f:
        f.write(f'idx,label,prediction,query\n')

    correct_ct = 0
    for idx, data in tqdm(enumerate(test_data)):
        query = data['query']
        gallery = data['gallery']
        label = data['label']
        # print(f'query: {query}')
        # print(f'label: {label}')

        # multi-image multi-round conversation, separate images
        query_value = load_image(os.path.join(data_folder, query), max_num=12).to(torch.bfloat16).cuda()
        gallery_values = []
        for i in range(5):
            gallery_values.append(load_image(os.path.join(data_folder, gallery[i]), max_num=12).to(torch.bfloat16).cuda())

        pixel_values = torch.cat([query_value] + gallery_values, dim=0)
        num_patches_list = [query_value.size(0)] + [v.size(0) for v in gallery_values]

        question = prompt_template
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=True)
        prediction = response.strip()

        if prediction == str(label):
            correct_ct += 1
        # print(f'User: {question}\nAssistant: {response}')
        # exit()

        # write out the label and prediction
        with open(save_filename, 'a') as f:
            f.write(f'{idx},{label},{prediction},{query}\n')

    print('Evaluation done!')
    # calculate accuracy
    accuracy = correct_ct / len(test_data) * 100
    print(f'Acc: {round(accuracy, 1)}')