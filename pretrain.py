import argparse
import os
import torch
import torch.distributed as dist

from src.common import PATHS
from src.tasks.pretrain_task import Pretraining
from src.utils.config import Config
from src.utils.tools import control_randomness, make_dir_if_not_exists, parse_config

def main_worker():
    # --------- Get Environment Variables ----------
    # local_rank = int(os.environ["LOCAL_RANK"])
    # global_rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    
    is_distributed = all(k in os.environ for k in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]) # 不强制读取了，万一不兼容分布式
    local_rank = int(os.environ["LOCAL_RANK"]) if is_distributed else 0
    global_rank = int(os.environ["RANK"]) if is_distributed else 0
    world_size = int(os.environ["WORLD_SIZE"]) if is_distributed else 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if is_distributed:
        backend = "nccl" if os.name != "nt" and torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    # --------- Load Config ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--patch_stride_len", type=int, default=None)
    parser.add_argument("--seq_len_channel", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    # parser.add_argument("--max_epochs", type=int, default=None) 你的pretrain.yaml里写的是max_epoch没有复数（捂脸
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--pos_embed_type", type=str, default=None)
    args_cmd = parser.parse_args()

    config = Config(
        config_file_path=args_cmd.config,
        default_config_file_path="configs/default.yaml"
    ).parse()

    control_randomness(config["random_seed"])

    config["device"] = local_rank
    config["rank"] = global_rank
    config["world_size"] = world_size
    config["distributed"] = is_distributed and world_size > 1
    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR
    ### Override config with command line arguments
    cli_overrides = {
        "patch_len": args_cmd.patch_len,
        "patch_stride_len": args_cmd.patch_stride_len,
        "seq_len_channel": args_cmd.seq_len_channel,
        "train_batch_size": args_cmd.train_batch_size,
        "val_batch_size": args_cmd.val_batch_size,
        "pos_embed_type": args_cmd.pos_embed_type,
        "max_epoch": args_cmd.max_epoch,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
    make_dir_if_not_exists(config["checkpoint_path"])

    args = parse_config(config)

    print(f"[Rank {global_rank}] Running with config:\n{args}\n")
    task_obj = Pretraining(args=args)

    if global_rank == 0:
        task_obj.setup_logger(notes="Pre-training runs")

    task_obj.train()

    if global_rank == 0:
        task_obj.end_logger()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main_worker()
