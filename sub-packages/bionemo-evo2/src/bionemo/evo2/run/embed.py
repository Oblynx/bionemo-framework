# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import functools
import tempfile
from pathlib import Path
from typing import Any, Literal

import nemo.lightning as nl
import torch
from lightning.pytorch import LightningDataModule
from megatron.core import parallel_state
from megatron.core.enums import Fp8Recipe
from megatron.core.utils import get_batch_on_this_cp_rank
from nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset import Evo2Dataset
from nemo.collections.llm.gpt.model.base import get_packed_seq_params
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.data import WrappedDataLoader
from nemo.utils import logging as logger

from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset

# Import Embedder classes
from bionemo.evo2.models.embedder import HyenaEmbedder, LlamaEmbedder, MambaEmbedder
from bionemo.evo2.models.llama import LLAMA_MODEL_OPTIONS
from bionemo.evo2.models.mamba import MAMBA_MODEL_OPTIONS
from bionemo.evo2.models.peft import Evo2LoRA
from bionemo.evo2.run.utils import infer_model_type, patch_eden_tokenizer
from bionemo.llm.data import collate
from bionemo.llm.utils.callbacks import PredictionWriter


CheckpointFormats = Literal["torch_dist", "zarr"]
PoolingStrategy = Literal["mean", "max", "last", "first", "per_token"]


def parse_args():
    """Parse arguments for Evo2 embedding extraction."""
    ap = argparse.ArgumentParser(description="Extract embeddings from Evo2 models.")
    ap.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use for prediction, defaults to 1.")
    ap.add_argument(
        "--devices",
        type=int,
        help="Number of devices to use for prediction, defaults to tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size.",
    )
    ap.add_argument(
        "--eden-tokenizer",
        action="store_true",
        help="Patch the tokenizer to work with the one used in training the Eden model.",
    )
    ap.add_argument("--fasta", type=Path, required=True, help="Fasta path from which to generate logit predictions.")
    ap.add_argument("--ckpt-dir", type=Path, required=True, help="NeMo2 checkpoint directory for inference.")
    ap.add_argument("--min-length", type=int, required=False, help="Minimum sequence length for padding.")
    ap.add_argument("--prepend-bos", action="store_true", help="Prepend BOS token to sequences. Defaults to False.")
    ap.add_argument(
        "--mask-phylogenetic-tags",
        action="store_true",
        help="Mask phylogenetic tags in loss computation. Defaults to False.",
    )
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1.")
    ap.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        choices=[1],
        default=1,
        help="Order of pipeline parallelism. Defaults to 1 and currently only 1 is supported.",
    )
    ap.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )
    ap.add_argument(
        "--fp8-recipe",
        type=str,
        default="delayed",
        choices=list(Fp8Recipe.__members__.keys()),
        help="FP8 recipe to use for FP8 tensors.",
    )
    ap.add_argument(
        "--no-sequence-parallel",
        action="store_true",
        help="When using TP, skip sequence parallelism.",
    )
    ap.add_argument("--micro-batch-size", type=int, default=1, help="Batch size for prediction. Defaults to 1.")
    ap.add_argument(
        "--write-interval",
        type=str,
        default="epoch",
        choices=["epoch", "batch"],
        help="Interval to write predictions to disk. If doing very large predictions, you may want to set this to 'batch'.",
    )
    ap.add_argument(
        "--model-size",
        type=str,
        default="7b_arc_longcontext",
        choices=sorted(
            list(HYENA_MODEL_OPTIONS.keys()) + list(MAMBA_MODEL_OPTIONS.keys()) + list(LLAMA_MODEL_OPTIONS.keys())
        ),
        help="Model size to use. Defaults to '7b_arc_longcontext'.",
    )
    # output args:
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir that will contain the generated text produced by the Evo2 model.",
    )
    ap.add_argument(
        "--files-per-subdir",
        type=int,
        help="Number of files to write to each subdirectory. If provided, subdirectories with N files each will be created. Ignored unless --write-interval is 'batch'.",
    )
    ap.add_argument(
        "--full-fp8",
        action="store_true",
        help="Use full FP8 precision (faster but less accurate) rather than vortex style which "
        "only applies FP8 to the projection layer of the hyena mixer, when using FP8.",
    )
    ap.add_argument("--fp8", action="store_true", help="Use FP8 precision. Defaults to BF16.")
    # extra:
    ap.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated.",
    )
    ap.add_argument(
        "--hybrid-override-pattern",
        type=str,
        help="Override the hybrid override pattern in the config (specifies hyena layer ordering and type).",
    )

    # Embedding-specific arguments
    ap.add_argument(
        "--embedding-layer",
        type=int,
        help="Extract embeddings from this layer (1-indexed). 0 for input embeddings. If not specified, uses last layer.",
    )
    ap.add_argument(
        "--pooling-strategy",
        type=str,
        choices=["mean", "max", "last", "first", "per_token"],
        default="mean",
        help="Pooling strategy for sequence embeddings.",
    )
    ap.add_argument(
        "--include-final-norm",
        action="store_true",
        default=True,
        help="Apply final layer norm to embeddings (default: True). Use --no-final-norm to disable.",
    )
    ap.add_argument(
        "--no-final-norm",
        action="store_false",
        dest="include_final_norm",
        help="Do not apply final layer norm to embeddings.",
    )

    ap.add_argument(
        "--seq-len-interpolation-factor",
        type=int,
        help="If set, override the sequence length interpolation factor specified in the requested config.",
    )
    ap.add_argument(
        "--lora-checkpoint-path",
        type=Path,
        required=False,
        default=None,
        help="Path to the lora states to restore from.",
    )
    return ap.parse_args()


class PredictDataModule(LightningDataModule):
    """Create a dataloader for prediction."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 1,
        tokenizer=None,
        min_length: int | None = None,
    ):
        """Create a dataloader for prediction."""
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.min_length = min_length
        default_pad_id = 0
        self.pad_token_id = getattr(tokenizer, "pad_id", default_pad_id) if tokenizer is not None else default_pad_id

    def setup(self, stage: str | None = None) -> None:
        """Set up the dataloader."""
        pass

    def predict_dataloader(self):
        """Create a dataloader for prediction."""
        # need to use this to communicate that we are in predict mode and safe to not drop last batch
        return WrappedDataLoader(
            mode="predict",
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=functools.partial(
                collate.padding_collate_fn,
                padding_values={"tokens": self.pad_token_id, "position_ids": self.pad_token_id, "loss_mask": False},
                min_length=self.min_length,
                max_length=None,
            ),
        )


def hyena_predict_forward_step(model, batch) -> torch.Tensor:
    """Performs a forward step for the Hyena model (used as dummy for config)."""
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
    }
    forward_args["attention_mask"] = None
    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)
    return model(**forward_args)


def hyena_predict_data_step(dataloader_iter) -> dict[str, torch.Tensor]:
    """Data step for the Hyena model prediction. Modified from the original gpt data step to include the seq_idx."""
    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_device_keys = set()
    required_host_keys = set()

    required_device_keys.add("attention_mask")
    if "cu_seqlens" in _batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    if parallel_state.is_pipeline_first_stage():
        required_device_keys.update(("tokens", "position_ids"))
    include_seq_idx = False
    if parallel_state.is_pipeline_last_stage():
        include_seq_idx = True
        required_device_keys.update(("labels", "tokens", "loss_mask"))

    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_cp_rank(_batch_required_keys)
    if include_seq_idx:
        output["seq_idx"] = _batch["seq_idx"].cuda(non_blocking=True)
    return output


def embed(
    fasta_path: Path,
    ckpt_dir: str,
    output_dir: Path,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    model_size: str,
    embedding_layer: int | None = None,
    pooling_strategy: PoolingStrategy = "mean",
    include_final_norm: bool = True,
    num_nodes: int = 1,
    devices: int | None = None,
    eden_tokenizer: bool = False,
    ckpt_format: CheckpointFormats = "torch_dist",
    fp8: bool = False,
    full_fp8: bool = False,
    fp8_recipe: str = "delayed",
    work_dir: Path | None = None,
    micro_batch_size: int = 1,
    write_interval: Literal["epoch", "batch"] = "epoch",
    prepend_bos: bool = False,
    no_sequence_parallel: bool = False,
    hybrid_override_pattern: str | None = None,
    seq_len_interpolation_factor: int | None = None,
    files_per_subdir: int | None = None,
    lora_checkpoint_path: Path | None = None,
    mask_phylogenetic_tags: bool = False,
    min_length: int | None = None,
    extra_callbacks: list | None = None,
):
    """Embedding extraction workflow for Evo2.

    Returns:
        None
    """
    if fp8 and not full_fp8 and fp8_recipe != "delayed":
        logger.warning(
            "fp8_recipe is ignored when using fp8 and not full_fp8 since it is set inside of the layer "
            "config to match vortex style FP8."
        )
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    if files_per_subdir is None and write_interval == "batch":
        logger.warning(
            "--files-per-subdir is not set with --write-interval batch, will write all predictions to a "
            "single directory. This may cause problems if you are predicting on a very large dataset."
        )
    sequence_parallel = tensor_parallel_size > 1 and not no_sequence_parallel
    output_dir.mkdir(parents=True, exist_ok=True)

    model_parallel_size = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    if devices is None:
        devices = model_parallel_size
    world_size = num_nodes * devices
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size must be divisible by model_parallel_size, got {world_size} and {model_parallel_size}."
        )
    global_batch_size = micro_batch_size * world_size // model_parallel_size

    # Configure PredictionWriter for embeddings
    # Using default key "embeddings" as per our design
    callbacks = [
        PredictionWriter(
            output_dir=output_dir,
            write_interval=write_interval,
            batch_dim_key_defaults={"embeddings": 0},
            seq_dim_key_defaults={"embeddings": 1},
            files_per_subdir=files_per_subdir,
            save_all_model_parallel_ranks=False,
        )
    ]
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    config_modifiers_init: dict[str, Any] = {
        "distribute_saved_activations": False if sequence_parallel and tensor_parallel_size > 1 else True,
    }
    if hybrid_override_pattern is not None:
        config_modifiers_init["hybrid_override_pattern"] = hybrid_override_pattern

    # Handle embedding_layer => num_layers truncation
    if embedding_layer is not None:
        # If user requests layer N (1-indexed), we need N layers in the model.
        if embedding_layer > 0:
            config_modifiers_init["num_layers"] = embedding_layer
        elif embedding_layer == 0:
            # Just embedding layer, no decoder layers needed?
            # We'll set num_layers=1 and just skip it in forward if needed, or rely on logic.
            # But usually config.num_layers must be >= 1 for validation.
            pass

    if seq_len_interpolation_factor is not None:
        config_modifiers_init["seq_len_interpolation_factor"] = seq_len_interpolation_factor

    tokenizer = get_nmt_tokenizer("byte-level")
    if eden_tokenizer:
        patch_eden_tokenizer(tokenizer)

    model_type = infer_model_type(model_size)

    # Select model config and Instantiate Embedder
    if model_type == "hyena":
        if model_size not in HYENA_MODEL_OPTIONS:
            raise ValueError(f"Invalid model size for Hyena: {model_size}")
        config = HYENA_MODEL_OPTIONS[model_size](
            forward_step_fn=hyena_predict_forward_step,
            data_step_fn=hyena_predict_data_step,
            vortex_style_fp8=fp8 and not full_fp8,
            **config_modifiers_init,
        )

        if lora_checkpoint_path:
            model_transform = Evo2LoRA(peft_ckpt_path=str(lora_checkpoint_path))
            callbacks.append(model_transform)
        else:
            model_transform = None

        model = HyenaEmbedder(
            config,
            tokenizer=tokenizer,
            embedding_layer=embedding_layer,
            pooling_strategy=pooling_strategy,
            include_final_norm=include_final_norm,
            model_transform=model_transform,
        )
    elif model_type == "mamba":
        if model_size not in MAMBA_MODEL_OPTIONS:
            raise ValueError(f"Invalid model size for Mamba: {model_size}")
        config = MAMBA_MODEL_OPTIONS[model_size](
            forward_step_fn=hyena_predict_forward_step,
            data_step_fn=hyena_predict_data_step,
            **config_modifiers_init,
        )

        model = MambaEmbedder(
            config,
            tokenizer=tokenizer,
            embedding_layer=embedding_layer,
            pooling_strategy=pooling_strategy,
            include_final_norm=include_final_norm,
        )
    elif model_type == "llama":
        if model_size not in LLAMA_MODEL_OPTIONS:
            raise ValueError(f"Invalid model size for Llama: {model_size}")

        config = LLAMA_MODEL_OPTIONS[model_size](
            forward_step_fn=hyena_predict_forward_step,
            data_step_fn=hyena_predict_data_step,
            **config_modifiers_init,
        )
        model = LlamaEmbedder(
            config,
            tokenizer=tokenizer,
            embedding_layer=embedding_layer,
            pooling_strategy=pooling_strategy,
            include_final_norm=include_final_norm,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}.")

    # Create PTL trainer.
    trainer = nl.Trainer(
        accelerator="gpu",
        num_nodes=num_nodes,
        devices=devices,
        strategy=nl.MegatronStrategy(
            drop_last_batch=False,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            sequence_parallel=sequence_parallel,
            save_ckpt_format=ckpt_format,
            ckpt_load_strictness="log_all",
            setup_optimizers=False,
            store_optimizer_states=False,
            configure_optimizers=False,
            data_sampler=nl.MegatronDataSampler(
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                seq_len=8192,
                output_log=False,
            ),
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            fp8_recipe=fp8_recipe,
            fp8="hybrid" if fp8 and full_fp8 else None,
            fp8_amax_history_len=16 if fp8 and full_fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 and full_fp8 else "most_recent",
        ),
    )

    nemo_logger = NeMoLogger(log_dir=str(work_dir))
    nemo_logger.setup(trainer, resume_if_exists=True)
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=False,
        resume_past_end=False,
        resume_from_path=str(ckpt_dir),
        restore_config=None,
    )

    resume.setup(trainer, model)

    if mask_phylogenetic_tags:

        def custom_loss_masker(tokens):
            return Evo2Dataset.mask_phylogenetic_tags(
                tokens,
                Evo2Dataset.TAG_BOUNDS,
                Evo2Dataset.TAG_CHARS,
                tokenizer.eod if tokenizer is not None else Evo2Dataset.DEFAULT_EOD,
                Evo2Dataset.MAX_TAG_LEN,
            )
    else:
        custom_loss_masker = None

    dataset = SimpleFastaDataset(fasta_path, tokenizer, prepend_bos=prepend_bos, custom_loss_masker=custom_loss_masker)
    datamodule = PredictDataModule(dataset, batch_size=micro_batch_size, tokenizer=tokenizer, min_length=min_length)
    trainer.predict(model, datamodule=datamodule)
    dataset.write_idx_map(output_dir)


def main():
    """Entrypoint for Evo2 embedding extraction."""
    args = parse_args()
    embed(
        num_nodes=args.num_nodes,
        devices=args.devices,
        fasta_path=args.fasta,
        ckpt_dir=args.ckpt_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        output_dir=args.output_dir,
        model_size=args.model_size,
        ckpt_format=args.ckpt_format,
        fp8=args.fp8,
        full_fp8=args.full_fp8,
        fp8_recipe=args.fp8_recipe,
        micro_batch_size=args.micro_batch_size,
        prepend_bos=args.prepend_bos,
        no_sequence_parallel=args.no_sequence_parallel,
        hybrid_override_pattern=args.hybrid_override_pattern,
        seq_len_interpolation_factor=args.seq_len_interpolation_factor,
        embedding_layer=args.embedding_layer,
        pooling_strategy=args.pooling_strategy,
        include_final_norm=args.include_final_norm,
        files_per_subdir=args.files_per_subdir,
        write_interval=args.write_interval,
        lora_checkpoint_path=args.lora_checkpoint_path,
        mask_phylogenetic_tags=args.mask_phylogenetic_tags,
        min_length=args.min_length,
        eden_tokenizer=args.eden_tokenizer,
    )


if __name__ == "__main__":
    main()
