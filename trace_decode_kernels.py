"""
Script to trace decode kernel usage in SGLang
"""

import argparse
import dataclasses
import time
import sys
import os
from datetime import datetime
import sglang as sgl
from sglang.srt.server_args import ServerArgs

class TeeOutput:
    """Class to write to both file and console"""
    def __init__(self, file_path, console_output=sys.stdout):
        self.file = open(file_path, 'w', buffering=1)  # Line buffered
        self.console = console_output
        
    def write(self, message):
        self.file.write(message)
        self.file.flush()  # Ensure immediate write
        # Only show ALEXANDER lines on console, others go to file only
        if "ALEXANDER" in message or "KERNEL CALL" in message or "INFERENCE COMPLETE" in message:
            self.console.write(message)
            
    def flush(self):
        self.file.flush()
        self.console.flush()
        
    def close(self):
        self.file.close()

def main():
    # Simple prompts to trigger decode kernels
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    
    # Short generation to focus on decode kernels
    sampling_params = {
        "temperature": 0.0,  # Deterministic for reproducible tracing
        "top_p": 1.0,
        "max_new_tokens": 4  # Generate a few tokens to see decode pattern
    }
    
    # Configure SGLang
    server_args = ServerArgs(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        disable_cuda_graph=True,  # Important: disable CUDA graph to see all calls
        mem_fraction_static=0.7,  # Adjust based on your GPU memory
    )
    
    # Set up logging to file
    log_file = "decode.log"
    print(f"Starting decode kernel tracing... Output will be saved to {log_file}")
    
    # Redirect stdout to capture all print statements
    original_stdout = sys.stdout
    tee_output = TeeOutput(log_file, original_stdout)
    sys.stdout = tee_output
    
    try:
        # Write header to log file
        print("="*100)
        print(f"DECODE KERNEL TRACE LOG - {datetime.now()}")
        print("="*100)
        print("")
        
        print("Starting SGLang engine...")
        llm = sgl.Engine(**dataclasses.asdict(server_args))
        
        print("Running inference with decode kernel tracing...")
        outputs = llm.generate(prompts, sampling_params)
        
        print("\n" + "="*80)
        print("INFERENCE COMPLETE - OUTPUTS:")
        print("="*80)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Generated: {output['text']}")
            print("-" * 40)
            
        print("\n" + "="*100)
        print(f"TRACE COMPLETE - {datetime.now()}")
        print("="*100)
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        tee_output.close()
        
        # Show summary to user
        print(f"\n✅ Decode kernel tracing complete!")
        print(f"📄 Full trace with paged attention metadata saved to: {log_file}")
        print(f"📊 Log file size: {os.path.getsize(log_file) / 1024:.1f} KB")
        
        # Show quick summary of what was captured
        with open(log_file, 'r') as f:
            content = f.read()
            alexander_lines = content.count("ALEXANDER")
            kernel_calls = content.count("KERNEL CALL START")
            
        print(f"📈 Captured {alexander_lines} debug lines and {kernel_calls} kernel calls")
        print(f"🔍 View the full trace: cat {log_file}")
        print(f"🔍 View just paged attention: grep 'ALEXANDER.*PAGED\\|KV_INDPTR\\|KV_INDICES' {log_file}")

if __name__ == "__main__":
    main()