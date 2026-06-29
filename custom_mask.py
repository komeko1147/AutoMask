
from __future__ import annotations

import argparse
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2_video_predictor


def _normalise_detections(result: dict, scores, labels) -> tuple[np.ndarray, np.ndarray, list[str]]:
    boxes = result["boxes"]
    if isinstance(boxes, torch.Tensor):
        box_values = boxes.detach().cpu().numpy()
    else:
        box_values = np.asarray(boxes, dtype=np.float32)

    if isinstance(scores, torch.Tensor):
        score_values = scores.detach().cpu().numpy()
    else:
        score_values = np.asarray(scores, dtype=np.float32)

    label_values = labels
    if isinstance(labels, torch.Tensor):
        label_values = labels.detach().cpu().tolist()
    elif isinstance(labels, np.ndarray):
        label_values = labels.tolist()

    return box_values, score_values, [str(label) for label in label_values]


def _box_area_ratio(box: np.ndarray, width: int, height: int) -> float:
    x0, y0, x1, y1 = box.tolist()
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return float(area / max(1.0, width * height))


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return inter / np.maximum(union, 1e-6)


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = _box_iou(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]
    return keep


def _select_boxes(
    result: dict,
    scores,
    labels,
    image_size: tuple[int, int],
    select_mode: str,
    max_objects: int,
    min_box_area_ratio: float,
    max_box_area_ratio: float,
    nms_iou_threshold: float,
) -> list[dict[str, object]]:
    """Select one or more GroundingDINO boxes and normalise them for SAM2."""

    width, height = image_size
    boxes, score_values, label_values = _normalise_detections(result, scores, labels)
    if len(boxes) == 0:
        return []

    candidates: list[int] = []
    for idx, box in enumerate(boxes):
        area_ratio = _box_area_ratio(box, width, height)
        if min_box_area_ratio <= area_ratio <= max_box_area_ratio:
            candidates.append(idx)

    if not candidates:
        return []

    candidate_boxes = boxes[candidates]
    candidate_scores = score_values[candidates]
    keep_local = _nms_indices(candidate_boxes, candidate_scores, nms_iou_threshold)
    keep = [candidates[idx] for idx in keep_local]
    keep.sort(key=lambda idx: float(score_values[idx]), reverse=True)
    if select_mode == "best":
        keep = keep[:1]
    else:
        keep = keep[:max_objects]

    selected = []
    for obj_id, idx in enumerate(keep, start=1):
        box = boxes[idx].astype(float)
        selected.append(
            {
                "obj_id": obj_id,
                "box": box.tolist(),
                "score": float(score_values[idx]),
                "label": label_values[idx],
                "area_ratio": _box_area_ratio(box, width, height),
            }
        )
    return selected


def _prepare_paths(video_root: Path, seq: str, save_root: Path) -> tuple[Path, Path]:
    seq_dir = video_root / seq
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence '{seq}' not found in {video_root}")
    save_dir = save_root / seq
    save_dir.mkdir(parents=True, exist_ok=True)
    return seq_dir, save_dir


def _discover_frames(seq_dir: Path) -> list[Path]:
    frames = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'}], key=lambda p: int(p.stem))
    if not frames:
        raise FileNotFoundError(f"No JPEG frames found in {seq_dir}")
    return frames


def _write_merged_masks(
    video_segments: dict[int, dict[int, np.ndarray]],
    frame_paths: list[Path],
    save_dir: Path,
    obj_ids: list[int],
    overwrite: bool,
) -> list[Path]:
    written = []
    for frame_idx, frame_path in enumerate(frame_paths):
        obj_map = video_segments.get(frame_idx, {})
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read frame for mask shape: {frame_path}")
        height, width = image.shape[:2]
        merged = np.zeros((height, width), dtype=bool)
        for obj_id in obj_ids:
            if obj_id not in obj_map:
                continue
            mask = np.asarray(obj_map[obj_id])
            mask = np.squeeze(mask)
            if mask.shape != merged.shape:
                raise ValueError(f"Mask shape mismatch for frame {frame_path.name}: {mask.shape} vs {merged.shape}")
            merged |= mask > 0
        frame_path = frame_paths[frame_idx]
        out_path = save_dir / f"dyn_mask_{frame_path.stem}.npz"
        if out_path.exists() and not overwrite:
            continue
        np.savez_compressed(out_path, dyn_mask=merged[np.newaxis, np.newaxis, ...].astype(np.uint8))
        written.append(out_path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate SAM2 masks for a sequence using a text prompt.")
    parser.add_argument('--video_dir', type=Path, required=True, help='Directory containing per-sequence image folders')
    parser.add_argument('--save_dir', type=Path, default=None, help='Output directory root for masks (defaults to video_dir/../sam_v2_dyn_mask)')
    parser.add_argument('--seq', type=str, required=True, help='Sequence name (subdirectory of video_dir)')
    parser.add_argument('--text_prompt', type=str, default='person', help='Text prompt for GroundingDINO')
    parser.add_argument('--box_threshold', type=float, default=0.4, help='GroundingDINO box confidence threshold')
    parser.add_argument('--text_threshold', type=float, default=0.3, help='GroundingDINO text confidence threshold')
    parser.add_argument('--model_id', type=str, default='IDEA-Research/grounding-dino-tiny', help='GroundingDINO model identifier')
    parser.add_argument('--sam_checkpoint', type=Path, default=Path('AutoMask/checkpoints/sam2_hiera_large.pt'), help='Path to SAM2 checkpoint')
    parser.add_argument('--sam_config', type=str, default='sam2_hiera_l.yaml', help='SAM2 config file name')
    parser.add_argument('--device', type=str, default='cuda', help='Torch device to run on')
    parser.add_argument('--select_mode', '--select-mode', choices=['best', 'all'], default='all', help='Use the best detection only or all selected detections')
    parser.add_argument('--max_objects', '--max-objects', type=int, default=20, help='Maximum number of detection boxes to seed into SAM2')
    parser.add_argument('--min_box_area_ratio', '--min-box-area-ratio', type=float, default=0.0005, help='Drop boxes smaller than this image-area ratio')
    parser.add_argument('--max_box_area_ratio', '--max-box-area-ratio', type=float, default=0.8, help='Drop boxes larger than this image-area ratio')
    parser.add_argument('--nms_iou_threshold', '--nms-iou-threshold', type=float, default=0.7, help='IoU threshold for detection NMS before SAM2 seeding')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing mask files')
    args = parser.parse_args(argv)

    if args.max_objects <= 0:
        raise ValueError("--max_objects must be positive")
    if args.min_box_area_ratio < 0.0 or args.max_box_area_ratio <= 0.0:
        raise ValueError("Box area ratios must be positive")
    if args.min_box_area_ratio > args.max_box_area_ratio:
        raise ValueError("--min_box_area_ratio cannot be greater than --max_box_area_ratio")
    if not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise ValueError("--nms_iou_threshold must be in [0, 1]")

    video_root = args.video_dir.resolve()
    save_root = args.save_dir if args.save_dir is not None else (video_root.parent / 'sam_v2_dyn_mask')
    save_root = save_root.resolve()

    seq_dir, seq_save_dir = _prepare_paths(video_root, args.seq, save_root)
    frame_paths = _discover_frames(seq_dir)

    if not args.overwrite:
        existing = list(seq_save_dir.glob('dyn_mask_*.npz'))
        if existing:
            print(f"[AutoMask] Found {len(existing)} mask files in {seq_save_dir}, skipping generation.")
            return

    print(frame_paths[0], 'frame_paths[0')
    first_frame = Image.open(frame_paths[0]).convert('RGB')

    processor = AutoProcessor.from_pretrained(args.model_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(args.device)

    inputs = processor(images=first_frame, text=[args.text_prompt], return_tensors='pt').to(args.device)
    with torch.no_grad():
        outputs = detector(**inputs)
    detections = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[first_frame.size[::-1]],
    )[0]

    if len(detections['boxes']) == 0:
        raise RuntimeError(f"No detections found for prompt '{args.text_prompt}' in {frame_paths[0]}")

    label_source = detections.get('text_labels', detections['labels'])
    selected_objects = _select_boxes(
        detections,
        detections['scores'],
        label_source,
        first_frame.size,
        args.select_mode,
        args.max_objects,
        args.min_box_area_ratio,
        args.max_box_area_ratio,
        args.nms_iou_threshold,
    )
    if not selected_objects:
        raise RuntimeError(
            f"No usable detections found for prompt '{args.text_prompt}' in {frame_paths[0]}; "
            "try lowering thresholds or relaxing box area filters."
        )
    print(
        f"[AutoMask] Selected {len(selected_objects)} / {len(detections['boxes'])} boxes "
        f"for prompt '{args.text_prompt}' (mode={args.select_mode})"
    )
    for obj in selected_objects:
        print(
            "[AutoMask] "
            f"obj_id={obj['obj_id']} label='{obj['label']}' score={obj['score']:.3f} "
            f"area={obj['area_ratio']:.4f} box={obj['box']}"
        )

    predictor = build_sam2_video_predictor(
        config_file=args.sam_config,
        ckpt_path=str(args.sam_checkpoint),
        device=args.device,
    )

    inference_state = predictor.init_state(video_path=str(seq_dir))
    obj_ids = [int(obj["obj_id"]) for obj in selected_objects]

    with torch.inference_mode(), torch.autocast(device_type='cuda' if args.device.startswith('cuda') else 'cpu', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
        for obj in selected_objects:
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(obj["obj_id"]),
                box=obj["box"],
            )
        video_segments: dict[int, dict[int, np.ndarray]] = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[idx] > 0.0).cpu().numpy()
                for idx, out_obj_id in enumerate(out_obj_ids)
            }

    written = _write_merged_masks(video_segments, frame_paths, seq_save_dir, obj_ids, args.overwrite)
    if not written:
        print(f"[AutoMask] No masks written for {args.seq}; consider using --overwrite.")
        return

    metadata = {
        'sequence': args.seq,
        'prompt': args.text_prompt,
        'select_mode': args.select_mode,
        'selected_object_count': len(selected_objects),
        'selected_objects': selected_objects,
        'box_threshold': args.box_threshold,
        'text_threshold': args.text_threshold,
        'max_objects': args.max_objects,
        'min_box_area_ratio': args.min_box_area_ratio,
        'max_box_area_ratio': args.max_box_area_ratio,
        'nms_iou_threshold': args.nms_iou_threshold,
        'mask_count': len(written),
    }
    with (seq_save_dir / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"[AutoMask] Saved {len(written)} masks to {seq_save_dir}")


if __name__ == '__main__':
    main()
