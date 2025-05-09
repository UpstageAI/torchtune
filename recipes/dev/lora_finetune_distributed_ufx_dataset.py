# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import time
import random
import gc
from functools import partial
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group

from torch.optim import Optimizer
from torchdata.nodes import (
    Loader, StopCriteria, IterableWrapper, ParallelMapper, SamplerWrapper
)
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path

from torchtune.data import padded_collate_packed
from torchtune.data._utils import chain, get_dataloader, get_multi_dataset
from torchtune.datasets._sft import SFTTransform
from torchtune.modules.peft import (
    DoRALinear,
    get_adapter_params,
    get_adapter_state_dict,
    get_lora_module_names,
    get_merged_lora_ckpt,
    LoRALinear,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DistributedSampler, Sampler

from tqdm import tqdm

log = utils.get_logger("DEBUG")

class DistributedBucketSampler(Sampler):
    """
    버킷 배칭 + 다중 노드 분할 샘플러
    - lengths: dataset['length']
    - batch_size: 노드당 batch size
    - world_size, rank: 분산 환경 정보
    - bucket_size: 버킷당 (world_size * batch_size * bucket_mul)
    - shuffle: epoch마다 셔플
    - seed: 기본 시드
    """
    def __init__(self,
                 lengths,
                 batch_size: int,
                 world_size: int,
                 rank: int,
                 bucket_size: int,
                 shuffle: bool = True,
                 seed: int = 0):
        self.lengths     = lengths
        self.batch_size  = batch_size
        self.world_size  = world_size
        self.rank        = rank
        self.bucket_size = bucket_size
        self.shuffle     = shuffle
        self.seed        = seed
        self.epoch       = 0
        self.indices     = list(range(len(lengths)))

    def __iter__(self):
        # 1) 길이 기준 정렬
        sorted_idx = sorted(self.indices, key=lambda i: self.lengths[i])
        # 2) 버킷으로 분할
        buckets = [
            sorted_idx[i : i + self.bucket_size]
            for i in range(0, len(sorted_idx), self.bucket_size)
        ]
        # 3) 버킷 순서 셔플
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(buckets)
        # 4) 각 버킷 내부 셔플
        for b in buckets:
            if self.shuffle:
                random.shuffle(b)
        # 5) 배치 단위로 쪼개기 → 완전한 배치만
        batches = []
        for b in buckets:
            for i in range(0, len(b), self.batch_size):
                batch = b[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
        # 6) 노드별 할당
        my_batches = batches[self.rank :: self.world_size]
        # 7) flatten해서 인덱스 단위로 yield
        for batch in my_batches:
            for idx in batch:
                yield idx

    def __len__(self):
        # 노드당 처리할 샘플 수
        total_batches = sum(
            (min(self.bucket_size, len(self.indices) - i) // self.batch_size)
            for i in range(0, len(self.indices), self.bucket_size)
        )
        my_batches = total_batches // self.world_size
        return my_batches * self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

class LoRAFinetuneRecipeDistributed(FTRecipeInterface):
    """
    Distributed LoRA finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - TorchData. Map and Streaming HuggingFace datasets, and multi-dataset mixing.
        - FSDP. Supported using PyTorch's FSDP APIs. CPU offload of parameters, gradients, and optimizer states
            is supported via ``fsdp_cpu_offload``. Resharding of parameters after the forward pass is
            done by default (corresponding to FULL_SHARD sharding strategy), but can be disabled by setting the config
            ``fsdp_reshard_after_forward`` to False (this corresponds to SHARD_GRAD_OP sharding strategy).
            DDP is currently not supported. Training on CPU is not supported.

        - Activation Checkpointing. This can be controlled using the ``enable_activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Activation Offloading. This can be controlled using the ``enable_activation_offloading``
            flag. Activation offloading is a technique similar to activations checkpointing that helps
            reduce the memory footprint to prevent OOMs on CUDA and enable bigger batches. Where activations
            checkpointing drops the activation in the forward to recompute it later in the backward,
            activations offloading will drop the activation in the forward to the CPU and bring it
            back during the backward pass. As always, there is a tradeoff--these savings in memory can
            come at the cost of training performance and CPU resources. To recover some runtime cost,
            we've added an option to enable offloading on a different stream to permit overlapping with
            the computation. This option is currently only available on PyTorch 2.5.0 or later and will be
            enabled by default if an acceptable torch version is found. Activation offloading can be used in
            conjunction with activation checkpointing.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Currently we checkpoint both the adapter weights (trainable params only) and the
            complete merged weights (adapter weights added back to the base model). For more details
            please take a look at our LoRA tutorial
            (https://pytorch.org/torchtune/main/tutorials/lora_finetune.html).

            Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training. Resuming
            training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/tutorials/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        ValueError: If world_size is 1
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
    """

    def __init__(self, cfg: DictConfig) -> None:
        device_type = cfg.device
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            device_type,
            offload_ops_to_cpu=self.fsdp_cpu_offload
            or self._enable_async_checkpointing,
        )
        init_process_group(self.distributed_backend)

        # Initialize distributed variables
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0

        self.tp_plan = config.instantiate(cfg.get("tensor_parallel_plan", None))
        self.tp_degree = cfg.get("tensor_parallel_dim", 1)
        if self.tp_degree > 1 and self.tp_plan is None:
            raise ValueError(
                "Tensor Parallel plan needs to be provided when tensor parallel is enabled."
            )
        data_shard = cfg.get("data_parallel_shard_dim", -1)  # -1 means to infer
        data_replicate = cfg.get("data_parallel_replicate_dim", 1)

        # Set up n-d device mesh
        self.parallel_dims = training.ParallelDims(
            dp_replicate=data_replicate,
            dp_shard=data_shard,
            tp=self.tp_degree,
            world_size=self.world_size,
        )
        self.world_mesh = self.parallel_dims.build_mesh(device_type=device_type)
        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            self.dp_degree, self.dp_rank = (
                dp_mesh.size(),
                dp_mesh.get_local_rank(),
            )
        else:
            self.dp_degree, self.dp_rank = 1, 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._batch_size = cfg.batch_size
        self._optimizer_in_bwd = cfg.get("optimizer_in_bwd", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        # Optimizer in backward is not compatible with gradient accumulation or gradient clipping
        if self._optimizer_in_bwd:
            if self._clip_grad_norm is not None:
                raise RuntimeError(
                    "Gradient clipping is not supported with optimizer in bwd."
                    "Please set clip_grad_norm=None, or optimizer_in_bwd=False."
                )
            if self._gradient_accumulation_steps > 1:
                raise RuntimeError(
                    "Gradient accumulation is not supported with optimizer in bwd."
                    "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
                )

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if device_type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )

        # These attributes constitute the recipe state and are updated by ``load_checkpoint``
        # when ``resume_from_checkpoint`` is ``True``
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self.train_samples_per_iter = self._batch_size * self._gradient_accumulation_steps * self.dp_degree


    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            should_load_recipe_state=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        # When resuming from checkpoint for LoRA, the recipe expects the adapter weights
        # and recipe state to be present. The keys should match up with what ``save_checkpoint``
        # used to create these intermediate checkpoints
        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not corrctly loaded
            # no need to check here
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        if self.fsdp_cpu_offload:
            # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
            # speed up when benchmarking fused AdamW on CPU
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            # log config with parameter override
            self._metric_logger.log_config(cfg)

        # Load the base model
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        self._compile = cfg.get("compile", False)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=self._optimizer_in_bwd,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )
        opt = self._optimizer  # 혹은 recipe._optimizer
        group = opt.param_groups[0]
        print("그룹당 파라미터 개수:", len(group["params"]))
        print("학습률(lr):", group["lr"])

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)
        utils.log_rank_zero(log, "Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._dataloader, self._len_datasets = self._setup_data(
            cfg_dataloader=cfg.dataloader,
            cfg_datasets=cfg.datasets,
            batch_size=cfg.batch_size,
            dataloader_state_dict=(
                checkpoint_dict[training.DATALOADER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = (
            self._len_datasets // self._gradient_accumulation_steps // self._batch_size
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        utils.log_rank_zero(
            log, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        base_model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
           c. We register (pre-)forward hooks with ``fully_shard`` instead of wrapping `nn.Module`
        """

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)

        utils.log_rank_zero(
            log,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # For FSDP sharding
        fsdp_shard_conditions = [
            partial(
                training.get_shard_conditions,
                names_to_match=custom_sharded_layers,
            )
        ]
        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
        )

        if lora_weights_state_dict:
            lora_missing, lora_unexpected = training.load_from_full_model_state_dict(
                model,
                lora_weights_state_dict,
                self._device,
                cpu_offload=fsdp_cpu_offload,
            )
        else:
            lora_missing, lora_unexpected = None, None

        # Initialize LoRA params and RoPE buffers
        with training.set_default_dtype(self._dtype), self._device:
            lora_device = "cpu" if fsdp_cpu_offload else self._device
            for m in model.modules():
                if (
                    isinstance(m, LoRALinear) or isinstance(m, DoRALinear)
                ) and not lora_weights_state_dict:
                    # lora may not be covered in state dict
                    # if finetune for the 1st time
                    m.to_empty(device=lora_device)
                    m.initialize_parameters()

                if hasattr(m, "rope_init"):
                    m.rope_init()

        base_missing, base_unexpected = training.load_from_full_model_state_dict(
            model,
            base_model_state_dict,
            self._device,
            cpu_offload=fsdp_cpu_offload,
        )

        for m in model.modules():
            if hasattr(m, "initialize_dora_magnitude"):
                m.initialize_dora_magnitude()

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            state_dict_keys=model.state_dict().keys(),
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        # log
        utils.log_rank_zero(
            log,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"전체 파라미터 수: {total_params:,}")
        print(f"훈련 가능한 파라미터 수: {trainable_params:,} {trainable_params/total_params*100:.2f}%")

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                param: config.instantiate(cfg_optimizer, [param])
                for param in self._model.parameters()
            }

            # Register optimizer step hooks on the model to run optimizer in backward.
            training.register_optim_in_bwd_hooks(
                model=self._model, optim_dict=optim_dict
            )
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
            # Load optimizer states for each param. If optimizer states are being restored in an optimizer in
            # backward run, these need to have been saved with the same setting. Cannot restore from runs that
            # did not use optimizer in backward.
            if opt_state_dict is not None:
                for param in opt_state_dict.keys():
                    try:
                        training.load_from_full_optimizer_state_dict(
                            self._model,
                            self._optim_ckpt_wrapper.optim_map[param],
                            opt_state_dict[param],
                            self._device,
                        )
                    except BaseException as e:
                        raise RuntimeError(
                            "Failed loading in-backward optimizer checkpoints."
                            "Please make sure run being restored from was using in-backward optimizer."
                        ) from e
            utils.log_rank_zero(log, "In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
            if opt_state_dict:
                training.load_from_full_optimizer_state_dict(
                    self._model,
                    optimizer,
                    opt_state_dict,
                    self._device,
                )

            utils.log_rank_zero(log, "Optimizer is initialized.")
            return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        """
        Set up the learning rate scheduler based on the provided configuration.
        It supports both standard optimization and optimizer-in-backward cases.

        Args:
            cfg_lr_scheduler (Optional[DictConfig]): The learning rate scheduler configuration.
            num_training_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch.

        Returns:
            lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                log.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        if self._optimizer_in_bwd:
            # Use the first optimizer from the wrapper to represent the learning rate
            optimizer = next(iter(self._optim_ckpt_wrapper.optim_map.values()))
        else:
            # Standard case: use the single optimizer
            optimizer = self._optimizer
        # Instantiate the learning rate scheduler
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        if self._optimizer_in_bwd:
            # Modify the scheduler for optimizer_in_bwd case
            self._optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataloader: DictConfig,
        cfg_datasets: ListConfig,
        batch_size: int,
        dataloader_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Loader:
        """
        Torchdata related setup happens here. Currently this recipe supports
        both Map and Streaming datasets (from HuggingFace datasets), and mixing multiple
        datasets (can be mix of Map and Streaming).
        """
        # Get global settings
        shuffle = cfg_dataloader.shuffle
        parallel_method = cfg_dataloader.get("parallel_method", "thread")
        packed = cfg_dataloader.get("packed", False)
        streaming = cfg_dataloader.get("streaming", False)
        num_workers = cfg_dataloader.get("num_workers", 0)
        pin_memory = cfg_dataloader.get("pin_memory", True)
        collate_fn = cfg_dataloader.collate_fn
        prefetch_factor = cfg_dataloader.get("prefetch_factor", 6)
        merge_datasets = cfg_dataloader.get("merge_datasets", False)

        if packed:
            raise ValueError("Packing not yet supported")

        # Multi-Dataset Stop Criteria
        stop_criteria = cfg_dataloader.get(
            "stop_criteria", StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED
        )
        weights, datasets = {}, {}
        len_datasets = 0
        dataset_list = []
        for idx, cfg_dataset in enumerate(cfg_datasets):
            dataset_name = cfg_dataset.pop("name", None)
            if dataset_name is None:
                dataset_name = cfg_dataset.get("subset", None)
            key = f"{idx}" + (f"_{dataset_name}" if dataset_name else "")

            utils.log_rank_zero(log, f"Instantiating dataset {cfg_dataset}")
            # Handle dataset-specific overrides, fallback to cfg_dataloader settings
            ds_streaming = cfg_dataset.pop("streaming", streaming)
            ds_shuffle = cfg_dataset.pop("shuffle", shuffle)
            ds_parallel_method = cfg_dataset.pop("parallel_method", parallel_method)
            ds_num_workers = cfg_dataset.pop("num_workers", num_workers)

            # Instantiate dataset transform
            assert "transform" in cfg_dataset, "transform must be specified in dataset"
            transform = config.instantiate(cfg_dataset.pop("transform"))

            weights[key] = float(cfg_dataset.pop("weight"))
            # datasets[key] = load_hf_dataset(
            #     **cfg_dataset,
            #     transform=transform,
            #     streaming=ds_streaming,
            #     shuffle=ds_shuffle,
            #     parallel_method=ds_parallel_method,
            #     num_workers=ds_num_workers,
            # )
            # Load a HuggingFace dataset (Map or Streaming) and apply a Transform to it.
            if "subset" in cfg_dataset:
                assert (
                    "name" not in cfg_dataset
                ), f"found both 'subset' and 'name' found, you may only specify one, {cfg_dataset=}"
                cfg_dataset["name"] = cfg_dataset.pop("subset")
            if "source" in cfg_dataset:
                source = cfg_dataset.pop("source")
            else:
                raise ValueError("source must be specified in dataset")
            # check if source is local path
            if os.path.exists(source):
                dataset = load_from_disk(source)
            else:
                dataset = load_dataset(source, **cfg_dataset)
            if "filter_fn" in cfg_dataset:
                dataset = dataset.filter(cfg_dataset.pop("filter_fn"))

            ratio = cfg_dataset.get("ratio", 1.0)
            if ratio < 1.0:
                dataset = dataset.select(range(int(len(dataset) * ratio)))

            if merge_datasets:
                dataset_list.append(dataset)
                # last dataset
                if idx + 1 == len(cfg_datasets):
                    dataset = concatenate_datasets(dataset_list)
                else:
                    continue

            world_size, rank = utils.get_world_size_and_rank()
            len_dataset = len(dataset) // world_size
            if ds_streaming:
                dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
                if ds_shuffle:
                    dataset = dataset.shuffle(seed=self.seed)
                node = IterableWrapper(dataset)
            else:
                if cfg_dataloader.get("bucket_ratio", 0) > 0:
                    # 버킷 샘플러로 교체
                    bucket_size = world_size * batch_size * cfg_dataloader.get("bucket_ratio", 50)
                    sampler = DistributedBucketSampler(
                        lengths=dataset["length"],
                        batch_size=batch_size,
                        world_size=world_size,
                        rank=rank,
                        bucket_size=bucket_size,
                        shuffle=ds_shuffle,
                        seed=self.seed,
                    )
                else:
                    sampler = DistributedSampler(
                        dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=ds_shuffle,
                        seed=self.seed,
                    )

                # Note: SamplerWrapper will call set_epoch on the sampler (if defined),
                # and auto-increment the epoch each time the node is reset.
                node = SamplerWrapper(sampler)
                transform = chain(dataset.__getitem__, transform)  # type: ignore

            node = ParallelMapper(
                node, map_fn=transform, num_workers=ds_num_workers, method=ds_parallel_method
            )
            datasets[key] = node
            len_datasets += int(len_dataset * weights[key])

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")

        collate_fn = (
            partial(
                _get_component_from_path(collate_fn),
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            )
            if not packed
            else padded_collate_packed
        )
        if len(datasets) > 1:
            dataset = get_multi_dataset(
                datasets=datasets,
                weights=weights,
                stop_criteria=stop_criteria,
            )
        else:
            dataset = next(iter(datasets.values()))

        loader = get_dataloader(
            dataset=dataset,
            model_transform=SFTTransform(model_transform=self._tokenizer),
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=num_workers,
            parallel_method=parallel_method,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )

        utils.log_rank_zero(log, "TorchData nodes are initialized")

        return loader, len_datasets

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete
        - If the `self._save_adapter_weights_only` option is True, the checkpointer will save only the adapter weights

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights.
        """
        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs

        utils.log_rank_zero(
            log,
            "Saving checkpoint. This may take some time. Retrieving full model state dict...",
        )
        start = time.perf_counter()

        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        cpu_state_dict = training.gather_cpu_state_dict(
            self._model,
            self._is_rank_zero,
            device=self._device,
            adapter_weights_only=self._save_adapter_weights_only,
        )
        utils.log_rank_zero(
            log,
            f"Getting full model state dict took {time.perf_counter() - start:.2f} secs",
        )

        if intermediate_checkpoint:
            utils.log_rank_zero(log, "Retrieving optimizer state dict...")
            opt_state_dict = training.get_full_optimizer_state_dict(
                self._model,
                self._optimizer,
                self._is_rank_zero,
                device=self._device,
            )
            utils.log_rank_zero(
                log,
                f"Getting optimizer state dict took {time.perf_counter() - start:.2f} secs",
            )
        else:
            opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:
            start = time.perf_counter()

            if self._save_adapter_weights_only:
                adapter_state_dict = cpu_state_dict
            else:
                # Filter out the adapter keys and weights from the model state dict. These will
                # be saved separately
                adapter_state_dict = get_adapter_state_dict(cpu_state_dict)

                # merge the adapter weights and base weights to create the model checkpoint
                merged_state_dict = get_merged_lora_ckpt(
                    cpu_state_dict,
                    rank=self._lora_rank,
                    alpha=self._lora_alpha,
                )
                checkpoint_dict.update({training.MODEL_KEY: merged_state_dict})
            checkpoint_dict.update({training.ADAPTER_KEY: adapter_state_dict})

            # if training is in-progress, checkpoint the optimizer state and recipe state
            # as well.
            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.SEED_KEY: self.seed,
                        training.EPOCHS_KEY: self.epochs_run,
                        training.TOTAL_EPOCHS_KEY: self.total_epochs,
                        training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                        training.DATALOADER_KEY: self._dataloader.state_dict(),
                    }
                )

            adapter_config = {
                "r": self._lora_rank,
                "lora_alpha": self._lora_alpha,
                "target_modules": get_lora_module_names(
                    self._lora_attn_modules,
                    self._apply_lora_to_mlp,
                    self._apply_lora_to_output,
                ),
                "peft_type": "LORA",
            }
            checkpoint_dict.update({training.ADAPTER_CONFIG: adapter_config})

            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
                adapter_only=self._save_adapter_weights_only,
            )
            log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")

        torch.distributed.barrier()

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        # 1) iterator를 한 번만 만듭니다
        data_iter = iter(self._dataloader)
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch, disable=not (self.rank == 0))
            for idx in range(self._steps_per_epoch):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # 데이터가 끝나면 바로 빠져나가도 되고,
                    # 또는 원하는 대로 재시작/종료 처리
                    break

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()

                utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                num_tokens += current_num_tokens

                # Shape [b, s], needed for the loss not the model
                labels = batch.pop("labels")

                with self.activations_handling_ctx:
                    logits = self._model(**batch)

                if not isinstance(logits, list):
                    labels = labels.reshape(-1)
                    logits = logits.reshape(-1, logits.size(-1))

                # Compute loss
                # Loss is normalized by default so we multiply by the number of tokens
                # This way we can normalize by the total number of tokens if we're accumulating gradients
                if current_num_tokens != 0:
                    current_loss = self._loss_fn(logits, labels) * current_num_tokens
                else:
                    # 학습할 토큰이 없는 경우 모델 그래프와 연결된 dummy loss 발생: gradient hook을 보장합니다.
                    # logits이 tensor인지 list인지 확인해서 모두 합친 뒤 dummy loss 생성
                    if isinstance(logits, list):
                        # 리스트 안의 모든 텐서를 먼저 합산
                        dummy = sum(t.sum() for t in logits)
                    else:
                        dummy = logits.sum()
                    # 그래프는 연결하지만 실제 값은 0인 loss
                    current_loss = dummy * 0

                # free logits otherwise it peaks backward memory
                del logits

                running_loss += current_loss
                # # step 전후 파라미터 차이 체크
                # old = self._model.decoder.layers[0]._checkpoint_wrapped_module.attn.q_proj.lora_a.weight.clone().detach()

                current_loss.backward()
                # for name, p in self._model.decoder.named_parameters():
                #     if "lora_a" in name or "lora_b" in name:
                #         print(f"{name}: grad norm = {p.grad.norm().item():.3e}" if p.grad is not None else f"{name}: NO GRAD")

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    # Get total number of tokens across all ranks to normalize gradients
                    torch.distributed.all_reduce(num_tokens)
                    # This will ensure that the logged loss matches what we're optimizing
                    torch.distributed.all_reduce(running_loss)
                    # Manually scale the gradients from unnormalized loss by total # of tokens
                    # We multiply by world_size to undo FSDP2 gradient normalization.
                    if num_tokens != 0:
                        training.scale_grads(self._model, self.dp_degree / num_tokens)

                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        ).full_tensor()
                    self._optimizer.step()
                    # new = self._model.decoder.layers[0]._checkpoint_wrapped_module.attn.q_proj.lora_a.weight
                    # print("Δparam norm:", (new - old).norm().item())

                    self._optimizer.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    if num_tokens != 0:
                        loss_to_log = running_loss.detach().item() / num_tokens
                    else:
                        loss_to_log = 0

                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}", refresh=False
                    )
                    pbar.update(1)

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": num_tokens
                            / (time_per_step * self.world_size),
                            "train_samples_per_second(batch*grad_acc*dp_size)": self.train_samples_per_iter / time_per_step
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )

                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                if (
                    (idx + 1) // self._gradient_accumulation_steps
                ) == self.max_steps_per_epoch:
                    break

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)
            # memory free
            torch.cuda.empty_cache()
            gc.collect()


        self._profiler.stop()

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    # init_process_group("cuda:nccl,cpu:gloo")
    # if cfg.get("fsdp_cpu_offload", False):
    #     # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
    #     # speed up when benchmarking fused AdamW on CPU
    #     training.set_torch_num_threads()

    config.log_config(recipe_name="LoRAFinetuneRecipeDistributed", cfg=cfg)

    recipe = LoRAFinetuneRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
