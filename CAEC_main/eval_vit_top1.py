# eval_vit_top1.py
import os
import time
import argparse
import multiprocessing
from typing import Dict, Tuple, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

from vit_model import vit_base_patch16_224 as create_model


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageNetFolderDataset(torch.utils.data.Dataset):
    """
    ImageNet-1K 验证集文件夹读取器。

    支持两种类别索引方式：
    1. 默认：按文件夹名称排序后生成 label。
    2. 可选：使用 imagenet_class_index.json，把 wnid 文件夹映射到官方类别 index。

    推荐验证集结构：
    val/
      n01440764/
      n01443537/
      ...
    """
    def __init__(self, root_dir: str, transform=None, class_index_json: str = ""):
        self.root_dir = root_dir
        self.transform = transform
        self.class_index_json = class_index_json
        self.extensions = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".JPG", ".PNG", ".BMP")

        assert os.path.isdir(root_dir), f"Validation path does not exist: {root_dir}"

        self.classes = self._find_classes()
        self.class_to_idx = self._build_class_to_idx()
        self.samples = self._make_dataset()

    def _find_classes(self) -> List[str]:
        classes = []
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                classes.append(item)
        classes.sort()
        return classes

    def _build_class_to_idx(self) -> Dict[str, int]:
        # 默认方式：文件夹名排序作为类别顺序
        # 如果你的 ImageNet val 文件夹是标准 wnid 顺序，这通常是可行的。
        return {cls_name: i for i, cls_name in enumerate(self.classes)}

    def _make_dataset(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            label = self.class_to_idx[class_name]

            for filename in os.listdir(class_dir):
                if filename.endswith(self.extensions):
                    samples.append((os.path.join(class_dir, filename), label))

        return samples

    def __getitem__(self, index):
        path, label = self.samples[index]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load image: {path}, error: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)


def build_transform(preprocess: str):
    """
    构建验证集预处理。

    preprocess='inception':
        mean/std = 0.5/0.5，常用于 Google/JAX ViT 权重。

    preprocess='imagenet':
        mean/std = ImageNet 标准归一化，常用于 torchvision/timm 常规 CNN/部分 ViT 权重。
    """
    if preprocess == "inception":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif preprocess == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unsupported preprocess: {preprocess}")

    print(f"[PREPROCESS] Resize(256) + CenterCrop(224)")
    print(f"[PREPROCESS] mean={mean}, std={std}")

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def safe_torch_load(path: str):
    """
    兼容 PyTorch 2.6+ 的 checkpoint 加载。

    PyTorch 2.6 起 torch.load 默认 weights_only=True，
    某些旧权重可能需要 fallback 到 weights_only=False。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print("[WARN] torch.load(weights_only=True) failed.")
        print(f"[WARN] Error: {e}")
        print("[WARN] Try torch.load(weights_only=False). Only do this for trusted checkpoints.")
        return torch.load(path, map_location="cpu", weights_only=False)


def unwrap_checkpoint(ckpt):
    """
    有些 checkpoint 外面会包一层 model/state_dict/module。
    这里自动拆开。
    """
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"[CKPT] Found wrapper key: {key}")
                return ckpt[key]

    return ckpt


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str):
    """
    去掉 DataParallel 保存时常见的 module. 前缀。
    """
    keys = list(state_dict.keys())
    if len(keys) > 0 and all(k.startswith(prefix) for k in keys[: min(10, len(keys))]):
        print(f"[CKPT] Strip prefix: {prefix}")
        return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def inspect_checkpoint(state_dict: Dict[str, torch.Tensor]):
    """
    输出 checkpoint 的关键信息，用于判断是不是 ImageNet-1k 权重。
    """
    print("\n========== Checkpoint Inspection ==========")
    print("[CKPT] Example keys:", list(state_dict.keys())[:8])

    if "head.weight" in state_dict:
        head_shape = tuple(state_dict["head.weight"].shape)
        print(f"[CKPT] head.weight shape: {head_shape}")
        if len(head_shape) == 2:
            print(f"[CKPT] checkpoint num_classes inferred from head: {head_shape[0]}")
            if head_shape[0] == 1000:
                print("[CKPT] This checkpoint has a 1000-class head. It is compatible with ImageNet-1k.")
            elif head_shape[0] == 21843:
                print("[CKPT] This checkpoint has a 21843-class head. It is ImageNet-21k, not directly ImageNet-1k.")
            else:
                print("[CKPT] This checkpoint has a non-1000 head. Check class definition.")
    else:
        print("[CKPT] No head.weight found. This may be a feature/backbone checkpoint.")

    if "pre_logits.fc.weight" in state_dict:
        print(f"[CKPT] pre_logits.fc.weight shape: {tuple(state_dict['pre_logits.fc.weight'].shape)}")
    else:
        print("[CKPT] No pre_logits.fc.weight found.")

    print("==========================================\n")


def load_weights_smart(model: torch.nn.Module, weights_path: str, device: torch.device):
    """
    智能加载权重。

    规则：
    1. 如果 checkpoint 参数和当前模型参数名字、形状都一致，则加载。
    2. 如果 head.weight 是 [1000, 768] 且模型也是 [1000, 768]，一定保留并加载。
    3. 只有形状不匹配时才跳过对应层。
    """
    assert os.path.exists(weights_path), f"Weights file does not exist: {weights_path}"

    ckpt = safe_torch_load(weights_path)
    state_dict = unwrap_checkpoint(ckpt)
    state_dict = strip_prefix_if_present(state_dict, "module.")

    inspect_checkpoint(state_dict)

    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for k, v in state_dict.items():
        if k not in model_state:
            skipped.append((k, tuple(v.shape) if hasattr(v, "shape") else "no_shape", "unexpected_key"))
            continue

        if hasattr(v, "shape") and tuple(v.shape) != tuple(model_state[k].shape):
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue

        filtered_state[k] = v

    load_info = model.load_state_dict(filtered_state, strict=False)

    print("\n========== Load Result ==========")
    print(f"[LOAD] Loaded tensors: {len(filtered_state)}")
    print(f"[LOAD] Skipped tensors: {len(skipped)}")

    if len(skipped) > 0:
        print("[LOAD] First skipped tensors:")
        for item in skipped[:10]:
            print("  ", item)

    print(f"[LOAD] Missing keys count: {len(load_info.missing_keys)}")
    print(f"[LOAD] Unexpected keys count: {len(load_info.unexpected_keys)}")

    if "head.weight" in load_info.missing_keys or "head.bias" in load_info.missing_keys:
        print("[WARNING] head.weight/head.bias were NOT loaded. The classifier head is random.")
        print("[WARNING] Top-1 will be close to random unless you fine-tune the head.")
    else:
        print("[LOAD] head.weight/head.bias loaded successfully.")

    if "pre_logits.fc.weight" in load_info.missing_keys:
        print("[INFO] pre_logits missing is OK if your vit_base_patch16_224 model has representation_size=None.")

    print("=================================\n")

    model.to(device)
    return model


@torch.no_grad()
def evaluate_topk(model, data_loader, device, use_amp=False):
    """
    评估 Top-1 和 Top-5。
    同时统计端到端吞吐量，包括 DataLoader + 模型推理。
    """
    model.eval()

    total = 0
    correct1 = 0
    correct5 = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for images, labels in tqdm(data_loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)

        _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))

        correct1 += correct[:, :1].sum().item()
        correct5 += correct[:, :5].sum().item()
        total += labels.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    end_to_end_throughput = total / elapsed

    return top1, top5, end_to_end_throughput


@torch.no_grad()
def benchmark_model_only(model, device, batch_size=64, runs=80, warmup_ratio=0.25, use_amp=False):
    """
    只测模型纯推理吞吐量，不包含图片读取和预处理。
    """
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)

    warmup = int(runs * warmup_ratio)
    total = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for i in tqdm(range(runs), desc="Benchmarking model-only"):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            total = 0
            start = time.time()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            _ = model(x)

        total += batch_size

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return total / elapsed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default=r"D:\imagenet-1k\val",
                        help="ImageNet-1K validation folder path.")
    parser.add_argument("--weights", type=str,
                        default=r"E:\zp\vision_transformer\vit_base_patch16_224.pth",
                        help="Checkpoint path.")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--preprocess", type=str, default="inception",
                        choices=["inception", "imagenet"],
                        help="inception means mean/std=0.5/0.5; imagenet means standard ImageNet mean/std.")
    parser.add_argument("--amp", action="store_true",
                        help="Use autocast mixed precision for evaluation and benchmark.")
    parser.add_argument("--benchmark-runs", type=int, default=80)

    return parser.parse_args()


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("\n========== Runtime ==========")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[CUDA] {torch.cuda.get_device_name(device)}")
    print(f"[Torch] {torch.__version__}")
    print(f"[Workers] num_workers={args.num_workers}")
    print("=============================\n")

    val_transform = build_transform(args.preprocess)

    val_dataset = ImageNetFolderDataset(
        root_dir=args.data_path,
        transform=val_transform,
    )

    print("\n========== Dataset ==========")
    print(f"[DATA] path: {args.data_path}")
    print(f"[DATA] classes: {len(val_dataset.classes)}")
    print(f"[DATA] samples: {len(val_dataset.samples)}")
    print(f"[DATA] first 10 class folders: {val_dataset.classes[:10]}")
    print(f"[DATA] last 10 class folders: {val_dataset.classes[-10:]}")

    if len(val_dataset.classes) != 1000:
        print("[WARNING] The number of validation folders is not 1000.")

    if len(val_dataset.classes) > 0 and val_dataset.classes[0] != "n01440764":
        print("[WARNING] Your first class folder is not n01440764.")
        print("[WARNING] If folder order does not match ImageNet official class index, accuracy can be near random.")
    print("=============================\n")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    val_loader = DataLoader(val_dataset, **loader_kwargs)

    print("\n========== Model ==========")
    model = create_model(num_classes=args.num_classes)
    print(f"[MODEL] create_model: vit_base_patch16_224")
    print(f"[MODEL] num_classes: {args.num_classes}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] params: {total_params / 1e6:.2f}M")
    print("===========================\n")

    model = load_weights_smart(model, args.weights, device)

    top1, top5, e2e_tps = evaluate_topk(
        model=model,
        data_loader=val_loader,
        device=device,
        use_amp=args.amp,
    )

    model_tps = benchmark_model_only(
        model=model,
        device=device,
        batch_size=args.batch_size,
        runs=args.benchmark_runs,
        use_amp=args.amp,
    )

    print("\n========== Final Result ==========")
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")
    print(f"End-to-end throughput: {e2e_tps:.2f} images/sec")
    print(f"Model-only throughput: {model_tps:.2f} images/sec")
    print("==================================\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()