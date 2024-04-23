import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from omegaconf import OmegaConf
from cheetah.common.config import Config
from cheetah.common.registry import registry
from cheetah.conversation.conversation_llama2 import Chat, CONV_VISION

from cheetah.models import *
from cheetah.processors import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds(seed = 50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()

config = OmegaConf.load(args.cfg_path)
cfg = Config.build_model_config(config)
model_cls = registry.get_model_class(cfg.model.arch)
model = model_cls.from_config(cfg.model).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.preprocess.vis_processor.eval
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

def parse_json_file(file_path, start_idx=0, end_idx=None):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Set end_idx to the length of the data if it is not specified
    if end_idx is None or end_idx > len(data):
        end_idx = len(data)

    # List to store the dictionaries
    result_list = []

    # Process each item in the specified range and store in the list
    for item in data[start_idx:end_idx]:
        entry = {
            "directory": item.get('directory', 'No directory info'),
            "pos_idx": item.get('pos_idx', 'No pos_idx info'),
            "neg_idx": item.get('neg_idx', 'No neg_idx info'),
            "caption": item.get('caption', 'No caption info')
        }
        result_list.append(entry)

    return result_list

# Example usage
# Replace 'your_file_path.json' with the path to your JSON file
parsed_data = parse_json_file('/content/drive/MyDrive/final/valid_simple.json', 0, 3000)

json_file_path = '/content/drive/MyDrive/final/gpt4v_result_full.json'
count = 0
for dictionary in parsed_data:

  directory = dictionary["directory"]
  pos_idx = dictionary["pos_idx"]
  neg_idx = dictionary["neg_idx"]
  caption = dictionary["caption"]
  directory_path = os.path.join(image_set_path,directory)
  first_image = get_nth_image(directory_path,pos_idx)
  print(first_image)
  second_image = get_nth_image(directory_path,neg_idx)
  print(second_image)
  image_path1 = os.path.join(directory_path,first_image)
  image_path2 = os.path.join(directory_path,second_image)


  description = caption
  context = "<Img><HereForImage></Img> <Img><HereForImage></Img> <Img><HereForImage></Img> <Img><HereForImage></Img> Given the context description: ? "
  context = context + description + "Return choice which matches the description most" 
  raw_img_list = [image_path1, image_path2]
  print("Question: ", context)
  llm_message = chat.answer(raw_img_list, context)
  print("Answer: ", llm_message)

