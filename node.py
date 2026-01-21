import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

import copy
import torch
import numpy as np
from PIL import Image
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
import glob
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry


logger = logging.getLogger('comfyui_segment_anything')

model_cache = {}

sam_model_dir_name = "sams"
sam_model_list = {
    "sam_vit_h (2.56GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "sam_vit_l (1.25GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "sam_vit_b (375MB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    },
    "sam_hq_vit_h (2.57GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
    },
    "sam_hq_vit_l (1.25GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
    },
    "sam_hq_vit_b (379MB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"
    },
    "mobile_sam(39MB)": {
        "model_url": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"
    }
}

groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def list_files(dirpath, extensions=[]):
    return [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)) and f.split('.')[-1] in extensions]


def list_sam_model():
    return list(sam_model_list.keys())


def load_sam_model(model_name, cache=True, device="cuda"):
    global model_cache
    sam = model_cache.get(model_name, None)
    if sam is None:
        sam_checkpoint_path = get_local_filepath(
            sam_model_list[model_name]["model_url"], sam_model_dir_name)
        model_file_name = os.path.basename(sam_checkpoint_path)
        model_type = model_file_name.split('.')[0]
        if 'hq' not in model_type and 'mobile' not in model_type:
            model_type = '_'.join(model_type.split('_')[:-1])
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        sam.model_name = model_file_name
        if cache:
            model_cache[model_name] = sam
    sam.to(device=device)
    sam.eval()
    return sam


def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f'using extra model: {destination}')
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination


def load_groundingdino_model(model_name, cache=True, device="cuda"):
    from local_groundingdino.datasets import transforms as T
    from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
    from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
    from local_groundingdino.models import build_model as local_groundingdino_build_model

    global model_cache
    dino = model_cache.get(model_name, None)
    if dino is None:
        dino_model_args = local_groundingdino_SLConfig.fromfile(
            get_local_filepath(
                groundingdino_model_list[model_name]["config_url"],
                groundingdino_model_dir_name
            ),
        )
        if dino_model_args.text_encoder_type == 'bert-base-uncased':
            dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
        dino = local_groundingdino_build_model(dino_model_args)
        checkpoint = torch.load(
            get_local_filepath(
                groundingdino_model_list[model_name]["model_url"],
                groundingdino_model_dir_name,
            ),
        )
        dino.load_state_dict(local_groundingdino_clean_state_dict(
            checkpoint['model']), strict=False)
        if cache:
            model_cache[model_name] = dino
    dino.to(device=device)
    dino.eval()
    return dino


def list_groundingdino_model():
    return list(groundingdino_model_list.keys())


def groundingdino_predict(
    dino_model,
    image,
    prompt,
    threshold,
    device="cuda"
):
    def load_dino_image(image_pil):
        from local_groundingdino.datasets import transforms as T
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image.to(device)

    def get_grounding_output(model, image, caption, box_threshold, device):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        return boxes_filt.to(device)

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(
        dino_model, dino_image, prompt, threshold, device
    )
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(device)
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt


def create_pil_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        output_masks.append(Image.fromarray(np.any(mask, axis=0)))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_images.append(Image.fromarray(image_np_copy))
    return output_images, output_masks


def create_tensor_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.cpu().numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(
            Image.fromarray(image_np_copy))
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (output_images, output_masks)


def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)


def sam_segment(
    sam_model,
    image,
    boxes,
    device="cuda"
):
    if boxes.shape[0] == 0:
        return None
    sam_is_hq = False
    # TODO: more elegant
    if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
        sam_is_hq = True
    predictor = SamPredictorHQ(sam_model, sam_is_hq)
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes, image_np.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False)
    masks = masks.permute(1, 0, 2, 3).cpu().numpy()  # Move to CPU before converting to numpy
    return create_tensor_output(image_np, masks, boxes)


class GroundingDinoSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "sam_model": (list_sam_model(), {"default": "sam_vit_b (375MB)"}, ),
                "grounding_dino_model": (list_groundingdino_model(), ),
                "device_mode": (["Auto", "Prefer GPU", "CPU"],
                    {
                        "tooltip": "Auto: Only applicable when a GPU is available. It temporarily loads models into VRAM only when the detection function is used.\n"
                        "Prefer GPU: Tries to keep models on the GPU whenever possible. This can be used when there is sufficient VRAM available.\n"
                        "CPU: Always loads only on the CPU."
                    },
                ),
                "global_cache": ('BOOLEAN', {"default": True}),
                "keep_models_loaded": ("BOOLEAN", {"default": True}),
            }
        }
    CATEGORY = "Segment Anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")


    def main(self, image, prompt, threshold, sam_model, grounding_dino_model, device_mode, global_cache, keep_models_loaded):
        if device_mode == "Prefer GPU" and torch.cuda.is_available():
            device = "cuda"
        elif device_mode == "CPU":
            device = "cpu"
        else:  # "Auto" mode: Use GPU if available, else fallback to CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"

        def sam_model_loader(model_name, global_cache):
            sam_model = load_sam_model(model_name, cache=global_cache, device=device)
            return sam_model

        def groundingdino_model_loader(model_name, global_cache):
            dino_model = load_groundingdino_model(model_name, cache=global_cache, device=device)
            return dino_model

        res_images = []
        res_masks = []
        for item in image:
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes = groundingdino_predict(
                groundingdino_model_loader(grounding_dino_model, global_cache),
                item,
                prompt,
                threshold,
                device
            )
            if boxes.shape[0] == 0:
                break
            (images, masks) = sam_segment(
                sam_model_loader(sam_model, global_cache),
                item,
                boxes,
                device
            )
            res_images.extend(images)
            res_masks.extend(masks)
        if len(res_images) == 0:
            _, height, width, _ = image.size()
            empty_images = torch.zeros((1, height, width, 3), dtype=torch.float32, device="cpu")
            empty_masks = torch.zeros((1, height, width), dtype=torch.float32, device="cpu")
            return (empty_images, empty_masks)
        if not keep_models_loaded:
            offload_device = comfy.model_management.unet_offload_device()
            print("Offloading models...")
            groundingdino_model_loader(grounding_dino_model, global_cache).to(offload_device)
            sam_model_loader(sam_model, global_cache).to(offload_device)
            comfy.model_management.soft_empty_cache()
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))


class IsMaskEmptyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["boolean_number"]

    FUNCTION = "main"
    CATEGORY = "Segment Anything"

    def main(self, mask):
        return (torch.all(mask == 0).int().item(), )
