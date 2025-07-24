"""
Usage:
python3 kernel_extract.py --model meta-llama/Llama-3.1-8B-Instruct --disable-cuda-graph
"""

import argparse
import dataclasses
import time

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def main(
    server_args: ServerArgs,
):
    # Sample prompts.
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        # "Hello, my name is Charlie"
    ]
    # Create a sampling params object.
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 50}

    # Create an LLM.
    llm = sgl.Engine(**dataclasses.asdict(server_args))

    outputs = llm.generate(prompts, sampling_params)
    time.sleep(2)
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
    time.sleep(2)


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)