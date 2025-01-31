#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024, FDU
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the FDU nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os
import itertools
import warnings
import torch
import random
import copy
from torch.utils.data import DataLoader
import pandas as pd

from gpu import add_gpu_params, parse_gpu, distributed_opt, distributed_sync, cleanup
from optimizer import (
    create_optimizer_scheduler,
    add_optimizer_params,
    create_adam_optimizer_from_args,
)

from data_utils import FT_Dataset
from splitmodel import GPT2Config, GPT2LMModel_Server, GPT2LMModel_Client
from exp_utils import create_exp_dir

import loralib as lora

torch.set_printoptions(threshold=100000)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="PyTorch GPT2 ft script")

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument(
    "--train_data0", required=True, help="location of training data corpus"
)

parser.add_argument(
    "--train_data1", required=True, help="location of training data corpus"
)

parser.add_argument(
    "--train_data2", required=True, help="location of training data corpus"
)

parser.add_argument(
    "--valid_data", required=True, help="location of validation data corpus"
)

parser.add_argument(
    "--train_batch_size", type=int, default=8, help="training batch size"
)

parser.add_argument(
    "--valid_batch_size", type=int, default=4, help="validation batch size"
)

parser.add_argument(
    "--grad_acc", type=int, default=1, help="gradient accumulation steps"
)

parser.add_argument("--clip", type=float, default=0.0, help="gradient clip")

parser.add_argument(
    "--seq_len", type=int, default=512, help="number of tokens to predict."
)

parser.add_argument(
    "--model_card",
    default="gpt2.md",
    choices=["gpt2.sm", "gpt2.md", "gpt2.lg"],
    help="model names",
)

parser.add_argument(
    "--init_checkpoint", default=None, help="pretrained checkpoint path"
)

parser.add_argument("--fp16", action="store_true", help="train model with fp16")

parser.add_argument("--log_interval", type=int, default=100, help="log interval")

parser.add_argument("--eval_interval", type=int, default=2000, help="eval interval")

parser.add_argument("--save_interval", type=int, default=500, help="save interval")

parser.add_argument(
    "--work_dir",
    type=str,
    default=os.getenv("PT_OUTPUT_DIR", "gpt2_model"),
    help="working folder.",
)

parser.add_argument("--lora_dim", type=int, default=0, help="lora attn dimension")

parser.add_argument("--lora_alpha", type=int, default=128, help="lora attn alpha")

parser.add_argument(
    "--obj",
    default="clm",
    choices=["jlm", "clm"],
    help="language model training objective",
)

parser.add_argument(
    "--lora_dropout",
    default=0.0,
    type=float,
    help="dropout probability for lora layers",
)

parser.add_argument("--label_smooth", default=0.0, type=float, help="label smoothing")

parser.add_argument("--roll_interval", type=int, default=-1, help="rolling interval")

parser.add_argument(
    "--roll_lr", type=float, default=0.00001, help="rolling learning rate"
)

parser.add_argument("--roll_step", type=int, default=100, help="rolling step")

parser.add_argument(
    "--eval_epoch", type=int, default=1, help="eval per number of epochs"
)


# influence model, calculate the influence score between two samples.
def print_args(args):
    if args.rank == 0:
        print("=" * 100)
        for k, v in args.__dict__.items():
            print(f"        - {k} : {v}")
        print("=" * 100)


def save_checkpoint(w_glob_client, model_server, args, train_step, num_clients):
    if args.rank != 0:
        return

    model_state_dict = {}

    # rename the key in client model
    for key, value in w_glob_client.items():
        new_key = ""
        if key.startswith("transformer_Client"):
            new_key = key.replace("transformer_Client", "module.transformer")
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value

    # rename the key in server model
    for key, value in model_server.state_dict().items():
        new_key = ""
        # print(key)
        if key.startswith("module.transformer_Server"):
            new_key = key.replace("module.transformer_Server", "module.transformer")
        else:
            print(key)
            model_state_dict[key] = value

        if new_key.startswith("module.transformer.h."):
            parts = key.split(".")
            layer_idx = int(parts[3])
            new_key = ".".join(["module.transformer.h", str(layer_idx + 3)] + parts[4:])
            model_state_dict[new_key] = value
        else:
            model_state_dict[new_key] = value
    if args.model_card == "gpt2.md":
        model_path = os.path.join(
            "./trained_models/GPT2_M/e2e",
            f"model_sfl.{train_step}_r={args.lora_dim}_num={num_clients}_block=3_seed={args.random_seed}.pt",
        )
    if args.model_card == "gpt2.sm":
        model_path = os.path.join(
            "./trained_models/GPT2_S/e2e",
            f"model_sfl.{train_step}_r={args.lora_dim}_num={num_clients}_block=3_seed={args.random_seed}.pt",
        )
    print("saving checkpoint", model_path)
    torch.save({"model_state_dict": model_state_dict}, model_path)


def fed_avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step(
    _loss,
    optimizer_server,
    model_server,
    optimizer_client,
    _schedule,
    client_hidden_states,
    hidden_states,
    args,
    is_update=True,
):
    if args.fp16:
        with amp.scale_loss(_loss, optimizer_server) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    dfx_client = client_hidden_states.grad.clone().detach()

    if is_update and args.clip > 0:
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer_server), args.clip
            )
        else:
            torch.nn.utils.clip_grad_norm_(model_server.parameters(), args.clip)
    optimizer_server.step()
    optimizer_server.zero_grad()

    if _schedule is not None:
        _schedule.step()

    hidden_states.backward(dfx_client)
    optimizer_client.step()
    optimizer_client.zero_grad()


def evaluate(model_client, model_server, valid_loader,args):
    model_client.eval()
    model_server.eval()
    device = torch.device("cuda")
    model_server = model_server.to(device)

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value.to(device) for key, value in data.items()}

            _input = data["input"]
            _target = data["target"]
            _msk = data["mask"]

            hidden_states, presents, _ = model_client(_input)

            _, _loss = model_server(
                _input.shape, hidden_states, presents, lm_labels=_target, lm_mask=_msk
            )
            loss = _loss.mean()

            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print("eval samples:", idx, "loss:", loss.float())

        print("average loss", avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(
    model_client,
    model_server,
    client_models,
    optimizers,
    optimizer_server,
    scheduler_server,
    train_loader0,
    train_loader1,
    train_loader2,
    valid_loader,
    args,
    train_step=0,
    epoch=0,
):
    """
    Function to train and validate federated learning models.

    Args:
        model_Client (torch.nn.Module): Client-side GPT-2 model.
        model_Server (torch.nn.Module): Server-side GPT-2 model.
        client_models (list): List of client GPT-2 models.
        optimizers (list): Optimizers for client models.
        optimizer_Server (torch.optim.Optimizer): Optimizer for server model.
        scheduler_Server (torch.optim.lr_scheduler): Learning rate scheduler for server optimizer.
        train_loader0, train_loader1, train_loader2 (torch.utils.data.DataLoader): Train loaders for three clients.
        valid_loader (torch.utils.data.DataLoader): Validation loader.
        args (argparse.Namespace): Command-line arguments.
        train_step (int, optional): Current training step. Default is 0.
        epoch (int, optional): Current epoch. Default is 0.

    Returns:
        Train step
    """
    model_client.train()
    model_server.train()
    # Meter to average language model loss
    avg_lm_loss = AverageMeter()
    print("start to train the model................", epoch)
    log_start_time = time.time()

    # Meter to average language model loss
    best_val_ppl = None

    device = torch.device("cuda")
    train_loader0.sampler.set_epoch(epoch)

    # Initialize global client model
    net_glob_client = GPT2LMModel_Client(config)
    net_glob_client = net_glob_client.to(device)
    # Load weights to global client model
    net_glob_client.load_weight(state_dict)
    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(net_glob_client)
    net_glob_client.train()
    w_glob_client = net_glob_client.state_dict()

    # aggregate every 100 train_step
    aggregate_step = 100

    w_locals_client = []

    # get train data from different client train dataset
    for idx, data in enumerate(zip(train_loader0, train_loader1, train_loader2)):
        # The client interacts with the server in turn
        for i in range(num_clients):
            client_data = {key: value.to(device) for key, value in data[i].items()}

            _input = client_data["input"]
            _target = client_data["target"]
            _msk = client_data["mask"]

            client_models[i].train()

            _input = _input.to(device)

            hidden_states, presents, w_client = client_models[i](_input)
            train_step += 1

            if (train_step + num_clients) % aggregate_step <= num_clients:
                w_locals_client.append(copy.deepcopy(w_client))

            client_hidden_states = hidden_states.clone().detach().requires_grad_(True)

            _, _lm_loss = model_server(
                _input.shape,
                client_hidden_states,
                presents,
                lm_labels=_target,
                lm_mask=_msk,
                label_smooth=args.label_smooth,
            )

            _lm_loss = _lm_loss.mean()

            is_update = train_step % args.grad_acc == 0
            avg_lm_loss.update(_lm_loss.item())

            optimizer_step(
                _lm_loss / args.grad_acc,
                optimizer_server,
                model_server,
                optimizers[i],
                scheduler_server,
                client_hidden_states,
                hidden_states,
                args,
                is_update=is_update,
            )

            # aggregate  client LoRA model every 100 train_step
            if train_step % aggregate_step == 0:
                temp_dict = {}
                w_locals_client_lora = []  # only aggregate lora-values

                for w_client in w_locals_client:
                    for key, value in w_client.items():
                        if key.endswith("lora_A"):
                            temp_dict[key] = value
                        if key.endswith("lora_B"):
                            temp_dict[key] = value
                    w_locals_client_lora.append(copy.deepcopy(temp_dict))

                w_glob_client_lora = fed_avg(w_locals_client_lora)

                w_glob_client_lora_new = {}

                for key, value in w_glob_client_lora.items():
                    new_key = "transformer_Client." + key
                    w_glob_client_lora_new[new_key] = value

                for key, value in w_glob_client.items():
                    if key.endswith("lora_A"):
                        w_glob_client[key] = w_glob_client_lora_new[key]
                    if key.endswith("lora_B"):
                        w_glob_client[key] = w_glob_client_lora_new[key]

                net_glob_client.load_state_dict(w_glob_client)
                for client_model in client_models:
                    client_model.load_state_dict(w_glob_client)

                w_locals_client = []

            # Output the training process data
            if train_step % args.log_interval == 0:
                elapsed = time.time() - log_start_time
                lr = optimizer_server.param_groups[0]["lr"]
                log_str = (
                    f"| epoch {epoch:3d} step {train_step:>8d} | {idx*3 + 1:>6d} batches | "
                    f"lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | "
                    f"loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | "
                    f"ppl {math.exp(avg_lm_loss.avg):5.2f}"
                )

                log_list.append(log_str)

                if args.rank == 0:
                    print(log_str)
                log_start_time = time.time()
                avg_lm_loss.reset()

            # save checkpoint at each save_interval
            if train_step % args.save_interval == 0:
                save_checkpoint(
                    w_glob_client, model_server, args, train_step, num_clients
                )
            distributed_sync(args)

            if train_step % args.eval_interval == 0:
                eval_start_time = time.time()

                valid_loss, valid_ppl = evaluate(
                    net_glob_client, model_server, valid_loader, args
                )
                if best_val_ppl is None or valid_ppl < best_val_ppl:
                    best_val_ppl = valid_ppl

                log_str = (
                    f"| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | "
                    f"time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | "
                    f"valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} "
                )
                log_list.append(log_str)

                if args.rank == 0:
                    print("-" * 100)
                    print(log_str)
                    print("-" * 100)

                net_glob_client.train()
                model_server.train()
                distributed_sync(args)

            # Save training process
            if train_step == args.max_step:
                df = pd.DataFrame(log_list, columns=["Log"])
                df.to_excel(
                    f"{args.model_card} rank={args.lora_dim} num={num_clients} block=3 seed={args.random_seed}.xlsx",
                    sheet_name="Sheet1",
                    index=False,
                )
                break

    # Save the final checkpoint
    if train_step == args.max_step:
        save_checkpoint(w_glob_client, model_server, args, train_step, num_clients)
    distributed_sync(args)
    return train_step


if __name__ == "__main__":
    args = parser.parse_args()
    parse_gpu(args)
    print_args(args)

    if args.fp16:
        try:
            from apex import amp
        except Exception:
            warnings.warn("Could not import amp, apex may not be installed")

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    # Create train dataset and valid dataset
    train_data0 = FT_Dataset(
        args.train_data0,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )

    train_data1 = FT_Dataset(
        args.train_data1,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )

    train_data2 = FT_Dataset(
        args.train_data2,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )

    valid_data = FT_Dataset(
        args.valid_data,
        args.valid_batch_size,
        args.seq_len,
    )

    train_loader0 = DataLoader(
        train_data0,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data0, seed=args.random_seed, shuffle=True
        ),
    )

    train_loader1 = DataLoader(
        train_data1,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data1, seed=args.random_seed, shuffle=True
        ),
    )

    train_loader2 = DataLoader(
        train_data2,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data2, seed=args.random_seed, shuffle=True
        ),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.valid_batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(
            valid_data, seed=args.random_seed
        ),
    )

    if args.model_card == "gpt2.sm":
        config = GPT2Config(
            n_embd=768,
            n_layer=12,
            n_head=12,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == "gpt2.md":
        config = GPT2Config(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == "gpt2.lg":
        config = GPT2Config(
            n_embd=1280,
            n_layer=36,
            n_head=20,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    lm_net_Client = GPT2LMModel_Client(config)
    lm_net_Server = GPT2LMModel_Server(config)

    state_dict = torch.load(args.init_checkpoint)
    if args.init_checkpoint is not None:
        print("loading model pretrained weight.")
        lm_net_Client.load_weight(state_dict)
        lm_net_Server.load_weight(state_dict)

    lm_net_Client = lm_net_Client.cuda()
    lm_net_Server = lm_net_Server.cuda()

    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net_Client)
        lora.mark_only_lora_as_trainable(lm_net_Server)

    optimizer_Client = create_adam_optimizer_from_args(lm_net_Client, args)
    optimizer_Server = create_adam_optimizer_from_args(lm_net_Server, args)

    # nums of clients:
    num_clients = 3
    client_models = []
    optimizers = []

    # Create client models for different clients
    for i in range(num_clients):
        client_model = GPT2LMModel_Client(config)
        client_model.load_weight(state_dict)
        client_model = client_model.cuda()
        if args.lora_dim > 0:
            lora.mark_only_lora_as_trainable(client_model)
        optimizer = create_adam_optimizer_from_args(client_model, args)
        client_models.append(client_model)
        optimizers.append(optimizer)

    if args.max_step is None:
        args.max_step = (
            args.max_epoch * train_data0.num_batches * 3 + args.world_size - 1
        ) // args.world_size
        print("set max_step:", args.max_step)

    scheduler_Client = create_optimizer_scheduler(optimizer_Client, args)
    scheduler_Server = create_optimizer_scheduler(optimizer_Server, args)

    if args.fp16:
        lm_net_Client, optimizer_Client = amp.initialize(
            lm_net_Client, optimizer_Client, opt_level="O1"
        )
        lm_net_Server, optimizer_Server = amp.initialize(
            lm_net_Server, optimizer_Server, opt_level="O1"
        )
    lm_net_Client, optimizer_Client = distributed_opt(
        args, lm_net_Client, optimizer_Client, grad_acc=args.grad_acc
    )
    lm_net_Server, optimizer_Server = distributed_opt(
        args, lm_net_Server, optimizer_Server, grad_acc=args.grad_acc
    )

    log_list = []

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                model_client=lm_net_Client,
                model_server=lm_net_Server,
                client_models=client_models,
                optimizers=optimizers,
                optimizer_server=optimizer_Server,
                scheduler_server=scheduler_Server,
                train_loader0=train_loader0,
                train_loader1=train_loader1,
                train_loader2=train_loader2,
                valid_loader=valid_loader,
                args=args,
                train_step=train_step,
                epoch=epoch,
            )

            if train_step >= args.max_step or (
                args.max_epoch is not None and epoch >= args.max_epoch
            ):
                if args.rank == 0:
                    print("-" * 100)
                    print("End of training")
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print("-" * 100)
            print("Exiting from training early")

    distributed_sync(args)
    print("cleanup dist ...")
    cleanup(args)
