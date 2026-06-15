import argparse
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

os.environ.setdefault("HF_HOME", os.path.abspath(".cache/huggingface"))
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    os.path.abspath(".cache/sentence_transformers"),
)

from src.common import EVENT_MAP
from src.data.load_data import generate_dsp
from src.models.mm_encoder import MultiModalEncoder
from src.models.timeseries_encoders.base import BaseModel

# Define a small trainable head on top of frozen TRACE embedding
class SegmentProjectionHead(nn.Module):
    def __init__(self, text_dim, segment_dim):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, segment_dim),
            nn.GELU(),
            nn.LayerNorm(segment_dim),
        )
        self.segment_proj = nn.Sequential(
            nn.Linear(segment_dim, segment_dim),
            nn.GELU(),
            nn.LayerNorm(segment_dim),
        )
        self.log_temperature = nn.Parameter(torch.tensor(np.log(0.07), dtype=torch.float32))

    def forward(self, text_embeddings, segment_embeddings):
        text_z = F.normalize(self.text_proj(text_embeddings), dim=-1)
        segment_z = F.normalize(self.segment_proj(segment_embeddings), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=1.0) # learnable parameter for scaling similarity scores of text2window 
        return torch.einsum("bd,bwd->bw", text_z, segment_z)/ temperature
        # bwd: batch size; number of windows; dimension
        # output: batch_size*num_windows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small segment scorer on a frozen checkpoint"
    )
    parser.add_argument("--checkpoint", default="results/model_checkpoints/context_align/retriever_demo.pt")
    parser.add_argument("--mode", default="train", choices=["train", "eval"])
    parser.add_argument("--data_dir", default="dataset/retrieval")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output", default="results/segment_weak/segment_head.pt")
    parser.add_argument("--head_checkpoint", default=None, help="Segment head checkpoint to evaluate.")

    # adjust the following hyperparameters as needed
    parser.add_argument("--window_size", type=int, default=48)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--segment_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=64, help="Use 0 to train on the full split.")
    parser.add_argument("--seed", type=int, default=13) # I wonder why u like to use 13 instead of 42. Okay i just follow your settings
    return parser.parse_args()


def load_trace_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    args.model_name = "TraceEncoder"
    args.rank = 0
    args.world_size = 1
    args.distributed = False
    args.device = device

    ts_state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        clean_key = key.replace("module.", "")
        if clean_key.startswith("ts_encoder."):
            ts_state_dict[clean_key[len("ts_encoder.") :]] = value

    original_loader = BaseModel.load_pretrained_weights # keep the original loader 
    BaseModel.load_pretrained_weights = staticmethod(
        lambda *loader_args, **loader_kwargs: {"model_state_dict": ts_state_dict}
    )
    try:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            model = MultiModalEncoder(args).to(device)
    finally:
        BaseModel.load_pretrained_weights = original_loader # restore the original loader

    state_dict = {
        key.replace("module.", ""): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters(): 
        param.requires_grad = False # freeze the model
    return model, args


def cache_path_for(cache_split, data_dir, text_encoder_name, legacy_name): 
    encoder_short_name = text_encoder_name.split("/")[-1]
    split_path = Path(data_dir) / f"{cache_split}_{legacy_name}_{encoder_short_name}.pt"
    legacy_path = Path(data_dir) / f"{legacy_name}_{encoder_short_name}.pt"
    return split_path, legacy_path


def load_or_create_description_embeddings(df, split, data_dir, text_encoder_name, cache_split):
    descriptions = [generate_dsp(row) for row in df["description"]]
    split_path, legacy_path = cache_path_for(cache_split, data_dir, text_encoder_name, "description_emb")

    candidate_paths = [split_path]
    if split == "test":
        candidate_paths.append(legacy_path)

    for path in candidate_paths:
        if path.exists():
            embeddings = torch.load(path, map_location="cpu")
            if embeddings.shape[0] >= len(df):
                return embeddings[: len(df)].float(), descriptions

    from sentence_transformers import SentenceTransformer

    print(f"Generating {split} description embeddings with {text_encoder_name}...")
    model = SentenceTransformer(text_encoder_name, trust_remote_code=True)
    embeddings = model.encode(
        descriptions,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
    ).float()
    split_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, split_path)
    return embeddings, descriptions


def load_split(data_dir, split, text_encoder_name, max_samples):
    parquet_path = Path(data_dir) / f"{split}.parquet"
    df_full = pd.read_parquet(parquet_path)
    df = df_full
    if max_samples and max_samples > 0:
        df = df_full.iloc[:max_samples].reset_index(drop=True)
    cache_split = split if len(df) == len(df_full) else f"{split}_first{len(df)}"

    text_embeddings, descriptions = load_or_create_description_embeddings(
        df, split, data_dir, text_encoder_name, cache_split
    )
    if max_samples and max_samples > 0:
        text_embeddings = text_embeddings[: len(df)]

    samples = []
    for idx, row in df.iterrows():
        timeseries = np.load(io.BytesIO(row["timeseries"])).astype(np.float32)
        event = row["events"]
        label = -100 if event is None else int(event["event_type"])
        samples.append(
            {
                "index": idx,
                "timeseries": scale_timeseries(timeseries),
                "text_embedding": text_embeddings[idx],
                "label": label,
                "description": descriptions[idx],
            }
        )
    return samples


def scale_timeseries(timeseries): # standardize each channel across time, then restore channel-first layout
    scaler = StandardScaler()
    return scaler.fit_transform(timeseries.T).T.astype(np.float32)


def make_windows(timeseries, window_size, stride, seq_len): # align all samples to the expected sequence length
    if window_size > seq_len:
        raise ValueError(f"window_size={window_size} cannot exceed model seq_len={seq_len}")

    raw_length = min(timeseries.shape[1], seq_len)
    aligned = np.zeros((timeseries.shape[0], seq_len), dtype=np.float32)
    aligned[:, :raw_length] = timeseries[:, :raw_length]

    starts = list(range(0, seq_len - window_size + 1, stride)) # Build candidate window
    last_start = seq_len - window_size
    if starts[-1] != last_start:
        starts.append(last_start)

    segments, masks = [], []
    for start in starts:
        end = start + window_size
        segment = np.zeros((timeseries.shape[0], seq_len), dtype=np.float32)
        mask = np.zeros((timeseries.shape[0], seq_len), dtype=np.int64)
        segment[:, :window_size] = aligned[:, start:end]
        valid = max(min(end, raw_length) - start, 0)
        if valid == 0:
            valid = 1
        mask[:, :valid] = 1
        segments.append(segment)
        masks.append(mask)
    # Each window is copied to the front of a full-length input, and i add a mask that marks the valid windows
    return (
        torch.from_numpy(np.stack(segments)),
        torch.from_numpy(np.stack(masks)),
        torch.tensor(starts, dtype=torch.long),
    ) 
    

def pseudo_segment_scores(timeseries, starts, window_size, label): # ai writes this heuristic scoring function
    # assign higher scores to windows  more likely to contain the event of interest, based on domain knowledge
    aligned_len = int(starts.max().item()) + window_size
    if timeseries.shape[1] < aligned_len:
        aligned = np.zeros((timeseries.shape[0], aligned_len), dtype=np.float32)
        aligned[:, : timeseries.shape[1]] = timeseries
        timeseries = aligned

    scores = []
    wind_mag = np.sqrt(timeseries[4] ** 2 + timeseries[5] ** 2)

    for start in starts.tolist():
        end = min(start + window_size, timeseries.shape[1])
        temp = timeseries[0, start:end]
        precip = timeseries[1, start:end]
        visibility = timeseries[3, start:end]
        wind = wind_mag[start:end]
        sky = timeseries[6, start:end]

        precip_score = float(np.nanmax(precip) + 0.5 * np.nanmean(precip) - 0.2 * np.nanmean(visibility))
        wind_score = float(np.nanmax(wind) + 0.5 * np.nanstd(wind))
        hail_score = float(np.nanmax(precip) + 0.2 * np.nanstd(temp) + 0.2 * np.nanmean(sky))
        generic_score = float(np.nanstd(timeseries[:, start:end]) + 0.1 * np.nanmax(np.abs(timeseries[:, start:end])))

        if label in [EVENT_MAP["Heavy Rain"], EVENT_MAP["Flash Flood"], EVENT_MAP["Flood"], EVENT_MAP["Debris Flow"]]:
            score = precip_score
        elif label in [EVENT_MAP["Thunderstorm Wind"], EVENT_MAP["Tornado"], EVENT_MAP["Funnel Cloud"], EVENT_MAP["Lightning"]]:
            score = wind_score + 0.25 * precip_score
        elif label == EVENT_MAP["Hail"]:
            score = hail_score
        else:
            score = generic_score
        scores.append(score)

    scores = torch.tensor(scores, dtype=torch.float32)
    if not torch.isfinite(scores).all() or float(scores.max() - scores.min()) < 1e-6:
        scores = torch.arange(len(scores), dtype=torch.float32)
    return scores


def encode_segments(trace_model, segments, masks, device, segment_batch_size): # encode all candidate windows for a sample in batches, then concatenate the embeddings
    outputs = []
    with torch.no_grad():
        for start in range(0, segments.size(0), segment_batch_size):
            end = start + segment_batch_size
            batch_segments = segments[start:end].to(device)
            batch_masks = masks[start:end].to(device)
            encoded = trace_model.get_ts_embedding(batch_segments, batch_masks).embeddings
            outputs.append(torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0).cpu())
    return torch.nan_to_num(torch.cat(outputs, dim=0), nan=0.0, posinf=0.0, neginf=0.0)


def train(args_cmd):
    torch.manual_seed(args_cmd.seed)
    np.random.seed(args_cmd.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trace_model, model_args = load_trace_model(args_cmd.checkpoint, device)
    seq_len = model_args.seq_len_channel
    dim = model_args.d_model

    samples = load_split(
        args_cmd.data_dir,
        args_cmd.split,
        model_args.text_encoder_name,
        args_cmd.max_samples,
    )
    samples = [sample for sample in samples if sample["label"] != -100]
    if not samples:
        raise RuntimeError("No labeled samples available for weak supervision.")

    text_dim = int(samples[0]["text_embedding"].numel())
    head = SegmentProjectionHead(text_dim, dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args_cmd.lr, weight_decay=1e-4)

    print(f"Frozen TRACE parameters: {sum(p.numel() for p in trace_model.parameters()):,}")
    print(f"Trainable segment-head parameters: {sum(p.numel() for p in head.parameters() if p.requires_grad):,}")
    print(f"Training samples: {len(samples)} | window={args_cmd.window_size} | stride={args_cmd.stride}")

    for epoch in range(args_cmd.epochs):
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch_start in tqdm(range(0, len(samples), args_cmd.batch_size), desc=f"Epoch {epoch + 1}/{args_cmd.epochs}"):
            batch = samples[batch_start : batch_start + args_cmd.batch_size]
            batch_segment_embeddings = []
            batch_text_embeddings = []
            targets = []

            for sample in batch:
                segments, masks, starts = make_windows(
                    sample["timeseries"],
                    args_cmd.window_size,
                    args_cmd.stride,
                    seq_len,
                )
                pseudo_scores = pseudo_segment_scores(
                    sample["timeseries"],
                    starts,
                    args_cmd.window_size,
                    sample["label"],
                )
                batch_segment_embeddings.append(
                    encode_segments(trace_model, segments, masks, device, args_cmd.segment_batch_size)
                )
                batch_text_embeddings.append(sample["text_embedding"])
                targets.append(int(torch.argmax(pseudo_scores).item()))

            segment_embeddings = torch.stack(batch_segment_embeddings).to(device)
            text_embeddings = torch.nan_to_num(
                torch.stack(batch_text_embeddings).to(device),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            target = torch.tensor(targets, dtype=torch.long, device=device)

            logits = head(text_embeddings, segment_embeddings)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total_count += len(batch)

        avg_loss = total_loss / total_count
        pseudo_acc = total_correct / total_count
        print(f"epoch={epoch + 1} loss={avg_loss:.4f} pseudo_target_match={pseudo_acc:.4f}")

    output = Path(args_cmd.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "config": vars(args_cmd),
            "model_dim": dim,
            "text_dim": text_dim,
            "seq_len": seq_len,
        },
        output,
    )

    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "output": str(output),
                "checkpoint": args_cmd.checkpoint,
                "split": args_cmd.split,
                "max_samples": args_cmd.max_samples,
                "window_size": args_cmd.window_size,
                "stride": args_cmd.stride,
                "frozen_trace": True,
                "trainable_parameters": sum(p.numel() for p in head.parameters() if p.requires_grad),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved weak segment head to {output}")
    print(f"Saved metadata to {metadata_path}")


def evaluate(args_cmd):
    torch.manual_seed(args_cmd.seed)
    np.random.seed(args_cmd.seed)

    if args_cmd.head_checkpoint is None:
        raise ValueError("--head_checkpoint is required when --mode eval")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trace_model, model_args = load_trace_model(args_cmd.checkpoint, device)
    seq_len = model_args.seq_len_channel
    dim = model_args.d_model

    samples = load_split(
        args_cmd.data_dir,
        args_cmd.split,
        model_args.text_encoder_name,
        args_cmd.max_samples,
    )
    samples = [sample for sample in samples if sample["label"] != -100]
    if not samples:
        raise RuntimeError("No labeled samples available for weak supervision.")

    text_dim = int(samples[0]["text_embedding"].numel())
    head = SegmentProjectionHead(text_dim, dim).to(device)
    checkpoint = torch.load(args_cmd.head_checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    total_correct = 0
    total_random_correct = 0
    total_count = 0
    total_windows = 0

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(samples), args_cmd.batch_size), desc=f"Eval {args_cmd.split}"):
            batch = samples[batch_start : batch_start + args_cmd.batch_size]
            batch_segment_embeddings = []
            batch_text_embeddings = []
            targets = []
            random_preds = []

            for sample in batch:
                segments, masks, starts = make_windows(
                    sample["timeseries"],
                    args_cmd.window_size,
                    args_cmd.stride,
                    seq_len,
                )
                pseudo_scores = pseudo_segment_scores(
                    sample["timeseries"],
                    starts,
                    args_cmd.window_size,
                    sample["label"],
                )
                num_windows = len(starts)
                batch_segment_embeddings.append(
                    encode_segments(trace_model, segments, masks, device, args_cmd.segment_batch_size)
                )
                batch_text_embeddings.append(sample["text_embedding"])
                targets.append(int(torch.argmax(pseudo_scores).item()))
                random_preds.append(int(torch.randint(num_windows, (1,)).item()))
                total_windows += num_windows

            segment_embeddings = torch.stack(batch_segment_embeddings).to(device)
            text_embeddings = torch.nan_to_num(
                torch.stack(batch_text_embeddings).to(device),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            target = torch.tensor(targets, dtype=torch.long, device=device)
            random_pred = torch.tensor(random_preds, dtype=torch.long, device=device)

            logits = head(text_embeddings, segment_embeddings)
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total_random_correct += (random_pred == target).sum().item()
            total_count += len(batch)

    pseudo_acc = total_correct / total_count
    random_acc = total_random_correct / total_count
    expected_random_acc = total_count / total_windows

    print(f"Evaluation split: {args_cmd.split}")
    print(f"Evaluation samples: {total_count} | window={args_cmd.window_size} | stride={args_cmd.stride}")
    print(f"pseudo_target_match: {pseudo_acc:.4f}")
    print(f"random_top1_acc_sampled: {random_acc:.4f}")
    print(f"random_top1_acc_expected: {expected_random_acc:.4f}")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
