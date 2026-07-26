from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """
    Represents a single pipeline stage.

    Attributes:
        stage_id: The ID of the pipeline stage.
        layers: List of layer indices assigned to this stage.
        device: The device (CPU/GPU) this stage runs on.
        rank: The process rank for distributed execution.
    """
    stage_id: int
    layers: list[int]
    device: torch.device
    rank: int


@dataclass
class MicroBatch:
    """
    Represents a micro-batch of data.

    Attributes:
        batch_id: The ID of the micro-batch.
        data: The tensor data for this micro-batch.
        is_last: Whether this is the last micro-batch in the batch.
    """
    batch_id: int
    data: torch.Tensor
    is_last: bool


class PipelineScheduler:
    """
    Implements 1F1B (one-forward-one-backward) pipeline scheduling for inference.

    For inference, 1F1B reduces to staggered forward passes across stages,
    but the term is retained for consistency with training schedules.
    """

    def __init__(self, num_stages: int, num_micro_batches: int, stage_id: int):
        """
        Initializes the pipeline scheduler.

        Args:
            num_stages: Total number of pipeline stages.
            num_micro_batches: Number of micro-batches to process.
            stage_id: The ID of the current pipeline stage.
        """
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.stage_id = stage_id

    def get_schedule(self) -> list[tuple[str, int]]:
        """
        Returns an ordered list of (action, micro_batch_id) tuples.
        Actions are: 'forward', 'send', 'recv'

        Returns:
            List of scheduling instructions.
        """
        schedule = []
        for mb_id in range(self.num_micro_batches):
            if self.stage_id > 0:
                schedule.append(('recv', mb_id))

            schedule.append(('forward', mb_id))

            if self.stage_id < self.num_stages - 1:
                schedule.append(('send', mb_id))

        return schedule


class PipelineStageManager:
    """Manages a single pipeline stage's execution."""

    def __init__(
        self,
        model_layers: nn.ModuleList,
        stage_id: int,
        num_stages: int,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None
    ):
        """
        Initializes the stage manager.

        Args:
            model_layers: The layers assigned to this stage.
            stage_id: The ID of the current pipeline stage.
            num_stages: Total number of pipeline stages.
            device: The device to run computations on.
            process_group: The distributed process group.
        """
        self.model_layers = model_layers.to(device)
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.device = device
        self.process_group = process_group

        self.active_sends = []
        self.active_recvs = []
        self._forward_time = 0.0
        self._bubble_time = 0.0

    def forward(self, micro_batch: MicroBatch) -> torch.Tensor:
        """
        Runs the assigned layers on the micro-batch data.

        Args:
            micro_batch: The micro-batch to process.

        Returns:
            The output tensor after processing through all stage layers.
        """
        x = micro_batch.data.to(self.device)
        for layer in self.model_layers:
            x = layer(x)
        return x

    def send_forward(self, tensor: torch.Tensor, dst_rank: int):
        """
        Sends activations to the next stage.

        Args:
            tensor: The tensor to send.
            dst_rank: The rank of the destination process.
        """
        if self.process_group is not None:
            # isend returns a work object, we should keep track of it
            work = dist.isend(tensor, dst=dst_rank, group=self.process_group)
            self.active_sends.append(work)

    def recv_forward(self, src_rank: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """
        Receives activations from the previous stage.

        Args:
            src_rank: The rank of the source process.
            shape: Expected shape of the incoming tensor.
            dtype: Expected data type of the incoming tensor.

        Returns:
            The received tensor.
        """
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        if self.process_group is not None:
            work = dist.irecv(tensor, src=src_rank, group=self.process_group)
            work.wait()  # Wait for receive to complete before returning
        return tensor

    def get_stats(self) -> dict:
        """
        Returns pipeline bubbles and utilization metrics.

        Returns:
            Dictionary containing metrics.
        """
        # Placeholder calculation
        total_time = max(1e-9, self._forward_time + self._bubble_time)
        utilization = self._forward_time / total_time
        bubble_ratio = self._bubble_time / total_time
        return {
            "utilization": utilization,
            "bubble_ratio": bubble_ratio
        }


class PipelineParallelEngine:
    """Orchestrates the full pipeline parallel inference."""

    def __init__(
        self,
        model: nn.Module,
        num_stages: int,
        rank: int,
        world_size: int,
        process_group: dist.ProcessGroup | None = None
    ):
        """
        Initializes the pipeline parallel engine.

        Args:
            model: The base model to partition.
            num_stages: Total number of pipeline stages.
            rank: The rank of the current process.
            world_size: Total number of processes.
            process_group: The distributed process group.
        """
        self.model = model
        self.num_stages = min(num_stages, world_size)
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group

        # Map rank to stage ID (simplest mapping: rank == stage_id for world_size == num_stages)
        self.stage_id = rank % self.num_stages

        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.stage_manager = self.partition_model()
        self.scheduler = PipelineScheduler(self.num_stages, num_micro_batches=1, stage_id=self.stage_id)

    def partition_model(self) -> PipelineStageManager:
        """
        Splits model layers evenly across stages.

        Returns:
            PipelineStageManager initialized with the assigned layers.
        """
        # Fallback if the model doesn't have a `.layers` ModuleList
        if not hasattr(self.model, 'layers') or not isinstance(getattr(self.model, 'layers', None), nn.ModuleList):
            logger.warning("Model does not have a 'layers' ModuleList attribute. Treating whole model as one layer.")
            layers = nn.ModuleList([self.model])
        else:
            layers = self.model.layers

        total_layers = len(layers)
        layers_per_stage = max(1, total_layers // self.num_stages)

        start_idx = self.stage_id * layers_per_stage

        # The last stage takes all remaining layers
        if self.stage_id == self.num_stages - 1:
            end_idx = total_layers
        else:
            end_idx = start_idx + layers_per_stage

        assigned_layers = nn.ModuleList([layers[i] for i in range(start_idx, end_idx)])

        return PipelineStageManager(
            model_layers=assigned_layers,
            stage_id=self.stage_id,
            num_stages=self.num_stages,
            device=self.device,
            process_group=self.process_group
        )

    def forward(
        self,
        input_tensor: torch.Tensor | None,
        num_micro_batches: int = 4,
        hidden_shape: tuple[int, ...] | None = None,
        hidden_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor | None:
        """
        Splits input into micro-batches, executes the 1F1B schedule,
        and gathers results on rank 0.

        Args:
            input_tensor: The input data (only needed on rank 0).
            num_micro_batches: Number of micro-batches to split into.
            hidden_shape: The expected shape of hidden states for receiving.
            hidden_dtype: The expected data type of hidden states.

        Returns:
            The final gathered output tensor on rank 0, or None on other ranks.
        """
        self.scheduler.num_micro_batches = num_micro_batches
        schedule = self.scheduler.get_schedule()

        micro_batches_data = []
        if self.stage_id == 0 and input_tensor is not None:
            # Chunk the tensor into micro-batches along the batch dimension
            micro_batches_data = list(torch.tensor_split(input_tensor, num_micro_batches, dim=0))
            if hidden_shape is None:
                # Infer hidden shape for intermediate communication based on first micro-batch
                # This is a simplification; actual shape depends on the model architecture.
                hidden_shape = micro_batches_data[0].shape

        output_micro_batches = {}
        activation_buffer = {}

        # Default hidden shape if not provided (e.g. for intermediate stages without input_tensor)
        if hidden_shape is None:
            hidden_shape = (1, 1024)

        for action, mb_id in schedule:
            if action == 'recv':
                src_rank = self.rank - 1
                tensor = self.stage_manager.recv_forward(src_rank, hidden_shape, hidden_dtype)
                activation_buffer[mb_id] = tensor

            elif action == 'forward':
                if self.stage_id == 0:
                    data = micro_batches_data[mb_id]
                else:
                    data = activation_buffer[mb_id]

                is_last = (mb_id == num_micro_batches - 1)
                mb = MicroBatch(batch_id=mb_id, data=data, is_last=is_last)

                out = self.stage_manager.forward(mb)
                activation_buffer[mb_id] = out

                # Update hidden shape for the next stage based on actual output
                hidden_shape = out.shape

                if self.stage_id == self.num_stages - 1:
                    output_micro_batches[mb_id] = out

            elif action == 'send':
                dst_rank = self.rank + 1
                tensor = activation_buffer[mb_id]
                self.stage_manager.send_forward(tensor, dst_rank)

        # Wait for all async sends to complete
        for work in self.stage_manager.active_sends:
            work.wait()
        self.stage_manager.active_sends.clear()

        # Gather outputs
        if self.stage_id == self.num_stages - 1:
            ordered_outputs = [output_micro_batches[i] for i in range(num_micro_batches)]
            final_output = torch.cat(ordered_outputs, dim=0)

            # Send back to rank 0 if the last stage is not on rank 0
            if self.rank != 0:
                if self.process_group is not None:
                    dist.send(final_output, dst=0, group=self.process_group)
                return None
            else:
                return final_output

        elif self.rank == 0 and self.stage_id != self.num_stages - 1:
            # Rank 0 needs to receive final output from the last stage
            last_rank = (self.num_stages - 1)
            # Simplification: assuming we know the final shape based on the input
            if input_tensor is not None:
                final_shape = input_tensor.shape
            else:
                final_shape = (1, 1024)

            final_tensor = torch.empty(final_shape, dtype=hidden_dtype, device=self.device)
            if self.process_group is not None:
                dist.recv(final_tensor, src=last_rank, group=self.process_group)
            return final_tensor

        return None

    def get_pipeline_stats(self) -> dict:
        """
        Returns utilization, bubble ratio, and other pipeline stats.

        Returns:
            Dictionary of statistics.
        """
        return self.stage_manager.get_stats()
