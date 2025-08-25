import json
import os
import inspect
import hashlib
from collections import defaultdict
from typing import Dict, Any
from safetensors.torch import save_file
from pathlib import Path
import torch

class KernelCallLogger_RecordTensor:
    def __init__(
        self,
        name: str,
        type: str,
        output_dir: str = "/home/akj2/kernel-tracing",
        environment: dict = None,
        tensor_format: str = "safetensors",
    ):
        self.name = name
        self.type = type
        self.env = environment or {}
        self.output_dir = Path(output_dir) / type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensor_format = tensor_format
        self.call_index = 0
        self.logged_axes = set()  # Track logged axes to avoid duplicates
        self.log_file = self.output_dir / f"{self.name}.workload.jsonl"

    def _generate_unique_id(self, axes: dict) -> str:
        """Generate a unique ID based on axes configuration."""
        # Create a consistent string representation of axes
        axes_str = json.dumps(axes, sort_keys=True)
        # Generate hash (first 32 chars like the example)
        return hashlib.md5(axes_str.encode()).hexdigest()

    def _tensor_save_path(self, axes: dict) -> str:
        """Generate descriptive filename for tensor storage."""
        unique_id = self._generate_unique_id(axes)
        # Use descriptive prefix based on kernel type
        if self.name == "batch_mla_paged_attention":
            prefix = "mla_meta"
        elif "decode" in self.type:
            prefix = "kv_meta"  
        else:
            prefix = "kernel_meta"
        return f"{prefix}_{unique_id}.{self.tensor_format}"

    def _axes_already_logged(self, axes: dict) -> bool:
        """Check if we've already logged a call with these exact axes values."""
        # Convert axes dict to a frozen representation for hashing
        axes_tuple = tuple(sorted(axes.items()))
        return axes_tuple in self.logged_axes

    def log_call(self, inputs: dict, axes: dict):
        # Skip if we've already logged this axes configuration
        if self._axes_already_logged(axes):
            return  # Skip duplicate
            
        # Define which inputs should be randomly generated vs stored
        data_tensors = {"q_nope", "q_pe", "ckv_cache", "kpe_cache"}  # Random generation
        # All other tensors (including 0-dim tensors like sm_scale, causal) get stored in safetensors
        
        input_specs = {}
        tensors_to_store = {}
        
        # Generate descriptive filename for this axes configuration
        tensor_filename = self._tensor_save_path(axes)
        tensor_filepath = self.output_dir / tensor_filename

        for name, tensor in inputs.items():
            if name in data_tensors:
                # Data tensors - just record as random with shape info
                input_specs[name] = {"type": "random"}
            else:
                tensors_to_store[name] = tensor.cpu().contiguous()
                input_specs[name] = {
                    "type": "safetensors",
                    "path": f"./{self.type}/{tensor_filename}",  # Relative path like the example
                    "tensor_key": name
                }

        # Save structural tensors to safetensors file if any exist
        if tensors_to_store:
            save_file(tensors_to_store, str(tensor_filepath))

        # Create workload entry in JSONL format
        workload_entry = {
            "definition": self.name,
            "solution": "",
            "workload": {
                "axes": axes,
                "inputs": input_specs
            },
            "evaluation": {}
        }

        # Write to JSONL file (append mode)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(workload_entry) + "\n")

        # Mark these axes as logged
        axes_tuple = tuple(sorted(axes.items()))
        self.logged_axes.add(axes_tuple)
        self.call_index += 1

    def save(self):
        # No need for separate save method since we write immediately to JSONL
        pass

import json
from pathlib import Path
from typing import Optional


class KernelCallLogger:
    def __init__(
        self,
        name: str,
        type: str,
        output_dir: str = "/home/akj2/kernel-tracing",
        environment: Optional[dict] = None,
    ):
        self.name = name
        self.type = type
        self.env = environment or {}
        self.output_dir = Path(output_dir) / type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.call_index = 0
        self.log_file = self.output_dir / f"{self.name}.workload.jsonl"

        # Write metadata header as first line
        header = {
            "name": self.name,
            "type": self.type,
            "environment": self.env,
            "format": "jsonl",
            "version": 1
        }
        with open(self.log_file, "w") as f:
            f.write(json.dumps(header) + "\n")

    def log_call(self, inputs: dict, axes: dict):
        input_shapes = {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype)
            }
            for name, tensor in inputs.items()
        }

        entry = {
            "axes": axes,
            "input_shapes": input_shapes
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self.call_index += 1