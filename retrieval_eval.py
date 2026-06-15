import argparse
import os
import time

os.environ.setdefault("HF_HOME", os.path.abspath(".cache/huggingface"))
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    os.path.abspath(".cache/sentence_transformers"),
)

import torch
import torch.nn.functional as F
from tqdm import tqdm
from rouge_score import rouge_scorer
from src.data.dataloader import get_dataloader
from src.models.mm_encoder import MultiModalEncoder
from src.models.timeseries_encoders.base import BaseModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation for TRACE."
    )
    parser.add_argument(
        "--checkpoint",
        default="results/model_checkpoints/context_align/retriever_demo.pt",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--query_idx", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "case", "table1"],
        help="case prints one retrieval example; table1 prints retrieval metrics.",
    )
    return parser.parse_args()


def load_model_and_data(args_cmd):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args_cmd.checkpoint,
        map_location=device,
        weights_only=False,
    )

    args = checkpoint["args"]
    args.model_name = "TraceEncoder"
    args.task_name = "retrieval"
    args.data_split = args_cmd.split
    args.batch_size = args_cmd.batch_size
    args.train_batch_size = args_cmd.batch_size
    args.val_batch_size = args_cmd.batch_size
    args.device = device
    args.distributed = False

    ts_state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        clean_key = key.replace("module.", "")  # using retriever_demo.pt instead of a not given checkpoint (swift-glitter-75/CATSEncoder.pth)
        if clean_key.startswith("ts_encoder."):
            ts_state_dict[clean_key[len("ts_encoder.") :]] = value

    # retriever_demo.pt already has ts_encoder.* weights. Here I just feed the encoder weights from checkpoint.
    original_loader = BaseModel.load_pretrained_weights
    BaseModel.load_pretrained_weights = staticmethod(
        lambda *loader_args, **loader_kwargs: {"model_state_dict": ts_state_dict}
    )
    try:
        model = MultiModalEncoder(args).to(device)
    finally: # to avoid side effects on other code
        BaseModel.load_pretrained_weights = original_loader
    state_dict = {
        key.replace("module.", ""): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    loader = get_dataloader(args)
    return device, model, loader


def encode_dataset(device, model, loader):
    ts_embeddings, text_embeddings, labels, timeseries = [], [], [], []
    descriptions, events = [], []
    encode_time = 0.0

    # no need to track gd during evaluation
    with torch.no_grad(): 
        for batch in tqdm(loader, desc="Encoding"):
            start = time.perf_counter()
            outputs = model(
                x_enc=batch.timeseries.float().to(device),
                input_mask=batch.input_mask.long().to(device),
                channel_description_emb=batch.channel_description_emb.to(device),
                description_emb=batch.description_emb.to(device),
                event_emb=batch.event_emb.to(device),
            )
            encode_time += time.perf_counter() - start

            ts_embeddings.append(outputs.embeddings.cpu())
            text_embeddings.append(outputs.description_emb.cpu())
            labels.append(torch.as_tensor(batch.labels).reshape(-1).cpu())
            timeseries.append(batch.timeseries.cpu())

            if hasattr(batch, "descriptions") and batch.descriptions is not None:
                descriptions.extend(batch.descriptions)
            if hasattr(batch, "events") and batch.events is not None:
                events.extend(batch.events)
 
    return { 
        "ts": F.normalize(torch.cat(ts_embeddings), dim=-1), 
        "text": F.normalize(torch.cat(text_embeddings), dim=-1),
        "labels": torch.cat(labels).long(),
        "timeseries": torch.cat(timeseries),
        "descriptions": descriptions,
        "events": events,
        "encode_time": encode_time,
    } # normalize for cosine similarity (?)


def label_metrics(similarity, query_labels, candidate_labels, ks=(1, 5)):
    # Precision over the top-k retrieved labels, not hit@k.
    order = torch.argsort(similarity, dim=1, descending=True)
    results = {}

    for k in ks:
        topk = order[:, :k]
        correct = candidate_labels[topk] == query_labels[:, None]
        results[f"P@{k}"] = correct.float().mean().item()

    matches = candidate_labels[order] == query_labels[:, None]
    reciprocal_ranks = []
    for row in matches:
        positions = torch.where(row)[0]
        reciprocal_ranks.append(
            0.0 if len(positions) == 0 else 1.0 / (positions[0].item() + 1)
        )
    results["MRR"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return results


def exact_pair_metrics(similarity, ks=(1, 5)): # modality matching
    order = torch.argsort(similarity, dim=1, descending=True)
    targets = torch.arange(similarity.shape[0])
    results = {}

    for k in ks:
        topk = order[:, :k]
        results[f"P@{k}"] = (topk == targets[:, None]).any(dim=1).float().mean().item()

    reciprocal_ranks = []
    for i, row in enumerate(order):
        positions = torch.where(row == i)[0]
        reciprocal_ranks.append(1.0 / (positions[0].item() + 1))
    results["MRR"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return results


def top1_similarity_stats(similarity, encoded):
    top1 = torch.argmax(similarity, dim=1)
    text_embeddings = encoded["text"]
    timeseries = encoded["timeseries"]

    text_cosine = (text_embeddings * text_embeddings[top1]).sum(dim=1).mean().item()
    ts_l1 = torch.abs(timeseries - timeseries[top1]).mean().item()
    ts_l2 = ((timeseries - timeseries[top1]) ** 2).mean().item()

    results = {
        "text_cosine": text_cosine,
        "ts_l1": ts_l1,
        "ts_l2": ts_l2,
    }

    descriptions = encoded["descriptions"]
    if descriptions:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = []
        for query_idx, candidate_idx in enumerate(top1.tolist()):
            score = scorer.score(
                descriptions[query_idx],
                descriptions[candidate_idx],
            )["rougeL"].fmeasure
            rouge_l.append(score)
        results["rougeL"] = sum(rouge_l) / len(rouge_l)

    return results


def print_case(encoded, query_idx, topk):
    ts_embeddings = encoded["ts"]
    text_embeddings = encoded["text"]
    descriptions = encoded["descriptions"]
    events = encoded["events"]

    if not descriptions:
        print("Train split has no raw text")
        return

    query_idx = min(query_idx, len(descriptions) - 1)

    # Compute similarity scores and retrieve top-k
    scores = text_embeddings[query_idx : query_idx + 1] @ ts_embeddings.T
    scores[0, query_idx] = -1e9
    values, indices = torch.topk(scores[0], k=topk)

    # print("\nExample retrieval")
    # print(f"\n[Query index] {query_idx}")
    # print("\n[Query Text]")
    # print(descriptions[query_idx])
    # print("\n[Query Event]")
    # print(events[query_idx])

    # print(f"\n[Retrieved Top-{topk}]")
    # for rank, (idx, score) in enumerate(zip(indices.tolist(), values.tolist()), 1):
    #     print(f"\nTop-{rank} | idx={idx} | score={score:.4f}")
    #     print("event:", events[idx])
    #     print("description:", descriptions[idx])


def print_results(encoded):
    ts_embeddings = encoded["ts"]
    text_embeddings = encoded["text"]
    labels = encoded["labels"]

    text_to_ts = text_embeddings @ ts_embeddings.T
    ts_to_text = ts_embeddings @ text_embeddings.T

    print("\nRetrieval metrics")
    print("\nLabel Matching")
    print("text2ts:", label_metrics(text_to_ts, labels, labels))
    print("ts2text:", label_metrics(ts_to_text, labels, labels))

    print("\nModality Matching")
    print("text2ts:", exact_pair_metrics(text_to_ts))
    print("ts2text:", exact_pair_metrics(ts_to_text))

    print("\nText/Time-Series Similarity")
    print("text2ts:", top1_similarity_stats(text_to_ts, encoded))
    print("ts2text:", top1_similarity_stats(ts_to_text, encoded))

    avg_query_time = encoded["encode_time"] / len(labels)
    print(f"\nAverage encoding time per query: {avg_query_time:.6f} seconds")


def main():
    args_cmd = parse_args()
    device, model, loader = load_model_and_data(args_cmd)
    encoded = encode_dataset(device, model, loader)

    if args_cmd.mode in ["all", "case"]:
        print_case(encoded, args_cmd.query_idx, args_cmd.topk)
    if args_cmd.mode in ["all", "table1"]:
        print_results(encoded)


if __name__ == "__main__":
    main()
