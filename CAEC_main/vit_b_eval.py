import os
import sys
import json
import time
import argparse
import subprocess
import multiprocessing
from typing import Dict, Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import timm

import tome


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageNetFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".JPG", ".PNG", ".BMP")

        assert os.path.isdir(root_dir), f"Validation path does not exist: {root_dir}"

        self.classes = self._find_classes()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _find_classes(self) -> List[str]:
        classes = []
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                classes.append(item)
        classes.sort()
        return classes

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
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print("[WARN] torch.load(weights_only=True) failed.")
        print(f"[WARN] Error: {e}")
        print("[WARN] Try weights_only=False. Only use trusted checkpoints.")
        return torch.load(path, map_location="cpu", weights_only=False)


def unwrap_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"[CKPT] Found wrapper key: {key}")
                return ckpt[key]
    return ckpt


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str):
    keys = list(state_dict.keys())
    if len(keys) > 0 and all(k.startswith(prefix) for k in keys[: min(10, len(keys))]):
        print(f"[CKPT] Strip prefix: {prefix}")
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def inspect_checkpoint(state_dict: Dict[str, torch.Tensor]):
    print("\n========== Checkpoint Inspection ==========")
    print("[CKPT] Example keys:", list(state_dict.keys())[:8])

    if "head.weight" in state_dict:
        head_shape = tuple(state_dict["head.weight"].shape)
        print(f"[CKPT] head.weight shape: {head_shape}")

        if len(head_shape) == 2:
            print(f"[CKPT] checkpoint num_classes inferred from head: {head_shape[0]}")
            if head_shape[0] == 1000:
                print("[CKPT] This checkpoint has a 1000-class head. ImageNet-1K compatible.")
            elif head_shape[0] == 21843:
                print("[CKPT] This checkpoint has a 21843-class head. ImageNet-21K.")
            else:
                print("[CKPT] Non-standard classifier head.")

    if "pre_logits.fc.weight" in state_dict:
        print(f"[CKPT] pre_logits.fc.weight shape: {tuple(state_dict['pre_logits.fc.weight'].shape)}")
    else:
        print("[CKPT] No pre_logits.fc.weight found.")
    print("==========================================\n")


def load_weights_smart(model: torch.nn.Module, weights_path: str, device: torch.device):
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
    print(f"[LOAD] Missing keys count: {len(load_info.missing_keys)}")
    print(f"[LOAD] Unexpected keys count: {len(load_info.unexpected_keys)}")

    if "head.weight" in load_info.missing_keys or "head.bias" in load_info.missing_keys:
        print("[WARNING] head.weight/head.bias were NOT loaded. The classifier head is random.")
    else:
        print("[LOAD] head.weight/head.bias loaded successfully.")
    print("=================================\n")

    model.to(device)
    return model


def build_model_for_method(args, method: str, device: torch.device):
    print(f"\n========== Build Model: {method} ==========")

    model = timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=args.num_classes
    )

    if method == "full_native":
        print("[METHOD] Full native timm ViT. No token merging. No ToMe patch.")

    elif method == "full_patched":
        print("[METHOD] Patched ViT with ToMeAttention, but r=0.")
        tome.patch.timm(
            model,
            trace_source=False,
            prop_attn=args.prop_attn,
            merge_method="tome",
        )
        model.r = 0

    elif method == "tome":
        print("[METHOD] Original ToMe BSM.")
        tome.patch.timm(
            model,
            trace_source=False,
            prop_attn=args.prop_attn,
            merge_method="tome",
        )
        model.r = args.r

    elif method == "ours":
        print("[METHOD] ETM-RL.")
        tome.patch.timm(
            model,
            trace_source=False,
            prop_attn=args.prop_attn,
            merge_method="etmrl",
        )
        model.r = args.r


    else:
        raise ValueError(f"Unsupported method: {method}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] model_name: {args.model_name}")
    print(f"[MODEL] num_classes: {args.num_classes}")
    print(f"[MODEL] params: {total_params / 1e6:.2f}M")

    if hasattr(model, "_tome_info"):
        print(f"[MODEL] merge_method: {model._tome_info.get('merge_method')}")
        print(f"[MODEL] prop_attn: {model._tome_info.get('prop_attn')}")
        print(f"[MODEL] r: {model.r}")
    else:
        print("[MODEL] no _tome_info, token merging disabled.")

    print("===========================================\n")

    model = load_weights_smart(model, args.weights, device)
    return model


@torch.no_grad()
def evaluate_topk(model, data_loader, device, use_amp=False):
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
    e2e_tps = total / elapsed

    return top1, top5, e2e_tps


@torch.no_grad()
def benchmark_model_only_cuda_event(
    model,
    device,
    batch_size=64,
    input_size=(3, 224, 224),
    warmup=50,
    runs=200,
    repeats=5,
    use_amp=False,
):
    """
    只测模型前向吞吐量，不包含 DataLoader、PIL 解码、图像预处理。

    使用 CUDA Event 计时，论文建议使用 median。
    """
    model.eval()
    x = torch.randn(batch_size, *input_size, device=device)

    throughputs = []

    for rep in range(repeats):
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            starter.record()
            for _ in range(runs):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    _ = model(x)
            ender.record()

            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender)
            elapsed_sec = elapsed_ms / 1000.0
        else:
            start = time.perf_counter()
            for _ in range(runs):
                _ = model(x)
            elapsed_sec = time.perf_counter() - start

        tps = batch_size * runs / elapsed_sec
        throughputs.append(tps)
        print(f"[BENCH] repeat {rep + 1}/{repeats}: {tps:.2f} images/sec")

    ts = torch.tensor(throughputs, dtype=torch.float32)
    return {
        "mean": float(ts.mean().item()),
        "median": float(ts.median().item()),
        "min": float(ts.min().item()),
        "max": float(ts.max().item()),
        "all": [float(v) for v in throughputs],
    }


def run_one_method(args, method: str):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("\n========== Runtime ==========")
    print(f"[METHOD] {method}")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[CUDA] {torch.cuda.get_device_name(device)}")
    print(f"[Torch] {torch.__version__}")
    print(f"[Workers] num_workers={args.num_workers}")
    print("=============================\n")

    val_transform = build_transform(args.preprocess)
    val_dataset = ImageNetFolderDataset(args.data_path, transform=val_transform)

    print("\n========== Dataset ==========")
    print(f"[DATA] path: {args.data_path}")
    print(f"[DATA] classes: {len(val_dataset.classes)}")
    print(f"[DATA] samples: {len(val_dataset.samples)}")
    print(f"[DATA] first 10 class folders: {val_dataset.classes[:10]}")
    print(f"[DATA] last 10 class folders: {val_dataset.classes[-10:]}")
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

    model = build_model_for_method(args, method, device)

    if args.skip_eval:
        top1, top5, e2e_tps = -1.0, -1.0, -1.0
    else:
        top1, top5, e2e_tps = evaluate_topk(
            model=model,
            data_loader=val_loader,
            device=device,
            use_amp=args.amp,
        )

    bench = benchmark_model_only_cuda_event(
        model=model,
        device=device,
        batch_size=args.batch_size,
        warmup=args.benchmark_warmup,
        runs=args.benchmark_runs,
        repeats=args.benchmark_repeats,
        use_amp=args.amp,
    )

    result = {
        "method": method,
        "r": 0 if method in ["full_native", "full_patched"] else args.r,
        "top1": top1,
        "top5": top5,
        "e2e_tps": e2e_tps,
        "model_tps_mean": bench["mean"],
        "model_tps_median": bench["median"],
        "model_tps_min": bench["min"],
        "model_tps_max": bench["max"],
        "model_tps_all": bench["all"],
    }

    print("\n========== Final Result ==========")
    print(f"Method: {result['method']}")
    print(f"r: {result['r']}")
    print(f"Top-1 Accuracy: {result['top1']:.2f}%")
    print(f"Top-5 Accuracy: {result['top5']:.2f}%")
    print(f"End-to-end throughput: {result['e2e_tps']:.2f} images/sec")
    print(f"Model-only throughput mean: {result['model_tps_mean']:.2f} images/sec")
    print(f"Model-only throughput median: {result['model_tps_median']:.2f} images/sec")
    print(f"Model-only throughput min/max: {result['model_tps_min']:.2f} / {result['model_tps_max']:.2f} images/sec")
    print("==================================\n")

    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))

    return result


def build_subprocess_command(args, method: str):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--method", method,
        "--data-path", args.data_path,
        "--weights", args.weights,
        "--model-name", args.model_name,
        "--num-classes", str(args.num_classes),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--r", str(args.r),
        "--preprocess", args.preprocess,
        "--benchmark-warmup", str(args.benchmark_warmup),
        "--benchmark-runs", str(args.benchmark_runs),
        "--benchmark-repeats", str(args.benchmark_repeats),
    ]

    if args.prop_attn:
        cmd.append("--prop-attn")
    if args.amp:
        cmd.append("--amp")
    if args.skip_eval:
        cmd.append("--skip-eval")

    return cmd


def run_all_by_subprocess(args):
    """
    all 模式用独立子进程分别运行，避免同一进程连续测试带来的 GPU/cache/函数切换污染。
    """
    methods = ["full_native", "full_patched", "tome", "ours"]
    results = []

    for method in methods:
        print(f"\n\n========== Launch subprocess for {method} ==========")
        cmd = build_subprocess_command(args, method)
        print("[CMD]", " ".join(cmd))

        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        print(proc.stdout)

        result = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                result = json.loads(line[len("RESULT_JSON:"):])
                break

        if result is None:
            raise RuntimeError(f"Failed to parse RESULT_JSON for method={method}")

        results.append(result)

    print("\n========== Subprocess Summary ==========")
    print(f"{'Method':<15} {'r':<5} {'Top-1':<10} {'Top-5':<10} {'E2E img/s':<14} {'Model median':<14} {'Model mean':<14}")
    for res in results:
        print(
            f"{res['method']:<15} "
            f"{res['r']:<5} "
            f"{res['top1']:<10.2f} "
            f"{res['top5']:<10.2f} "
            f"{res['e2e_tps']:<14.2f} "
            f"{res['model_tps_median']:<14.2f} "
            f"{res['model_tps_mean']:<14.2f}"
        )
    print("========================================\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="all",
                        choices=["full_native", "full_patched", "tome", "ours", "all"])

    parser.add_argument("--data-path", type=str, default=r"D:\imagenet-1k\val")
    parser.add_argument("--weights", type=str,
                        default=r"E:\zp\vision_transformer\vit_base_patch16_224.pth")
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--r", type=int, default=20)

    parser.add_argument("--preprocess", type=str, default="inception",
                        choices=["inception", "imagenet"])

    parser.add_argument("--prop-attn", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--skip-eval", action="store_true",
                        help="Only benchmark model-only throughput. Do not run ImageNet validation.")

    parser.add_argument("--benchmark-warmup", type=int, default=50)
    parser.add_argument("--benchmark-runs", type=int, default=200)
    parser.add_argument("--benchmark-repeats", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.method == "all":
        run_all_by_subprocess(args)
    else:
        run_one_method(args, args.method)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()