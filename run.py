import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import ray
import torch
import wandb
from datasets import load_dataset
from ray.util.placement_group import placement_group
from vllm import LLM, RequestOutput, SamplingParams

from routers import BERTRouter


JOB_ID = os.getenv("SLURM_JOB_ID", str(uuid4()))
NUM_CPU = os.cpu_count()
NUM_GPU = torch.cuda.device_count()

# GPUs
DEVICE = "cuda:0"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm-routing")

# Models
STRONG_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
WEAK_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Outputs
OUTPUTS = Path("outputs")
RESULTS = OUTPUTS / JOB_ID / "results.jsonl"

# Router
CHECKPOINT_PATH = "routellm/bert_gpt4_augmented"

# Random seed
SEED = 4013284336
random.seed(SEED)

TARGET_RATIO = 0.5


def save(l):
    """A utility function to save model results."""
    with open(RESULTS, "a") as f:
        j = json.dumps(l)
        f.write(j + "\n")


@ray.remote
class LLMActor:
    def __init__(
        self,
        model_name: str,
        device: str,
        gpu_memory_utilization: float,
        max_model_len: int = 512,
    ) -> None:
        self.model = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.device = device

    async def generate_text(
        self,
        prompt: str,
        top_p: float = 0.95,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> RequestOutput:
        sampling_params = SamplingParams(
            top_p=top_p, temperature=temperature, max_tokens=max_tokens
        )
        output = self.model.generate([prompt], sampling_params)
        return output


async def main():
    wandb.init(
        project="llm-routing",
        name=f"eli5-{JOB_ID}",
        config={
            "job_id": JOB_ID,
            "num_cpu": NUM_CPU,
            "num_gpu": NUM_GPU,
            "seed": SEED,
            "target_ratio": TARGET_RATIO,
        },
    )
    logger.info(f"Started job {JOB_ID} at {datetime.now().isoformat()}")

    ray.init(address="local")
    pg = placement_group([{"GPU": 0.5, "CPU": 1}, {"GPU": 0.5, "CPU": 1}])
    ray.get(pg.ready())

    strong_model = LLMActor.options(
        placement_group=pg,
        placement_group_bundle_index=0,
        num_gpus=0.5,
    ).remote(model_name=STRONG_MODEL, device=DEVICE, gpu_memory_utilization=0.65)

    weak_model = LLMActor.options(
        placement_group=pg,
        placement_group_bundle_index=1,
        num_gpus=0.5,
    ).remote(model_name=WEAK_MODEL, device=DEVICE, gpu_memory_utilization=0.25)

    router = BERTRouter(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)

    async def send_request_to_model(actor, prompt, top_p=0.95, temperature=0.8):
        result = await actor.generate_text.remote(
            prompt, top_p=top_p, temperature=temperature
        )
        return result

    async def process_query(
        query,
        strong_model_queue,
        weak_model_queue,
        threshold,
        learning_rate,
        target_ratio,
    ):
        # Compute current and target queue ratios
        strong_queue_depth = strong_model_queue.qsize()
        weak_queue_depth = weak_model_queue.qsize()
        total_queue_depth = strong_queue_depth + weak_queue_depth + 1e-6
        current_queue_ratio = strong_queue_depth / total_queue_depth
        error = current_queue_ratio - target_ratio

        # Adjust threshold
        threshold += learning_rate * error
        threshold = max(0, min(threshold, 1))
        logger.info(
            f"strong_queue_depth: {strong_queue_depth}. "
            f"weak_queue_depth: {weak_queue_depth}. "
            f"current_queue_ratio: {current_queue_ratio:.3f}. "
            f"target_ratio: {target_ratio:.3f}. "
            f"error: {error:.3f}. "
            f"threshold: {threshold:.3f}."
        )

        # Route query
        probability = router.calculate_strong_win_rate(query)
        if probability > threshold:
            logger.info(
                f"Routed query to strong model: {query} (p={probability}, "
                f"adjusted_threshold={threshold})"
            )
            await strong_model_queue.put(query)
        else:
            logger.info(
                f"Routed query to weak model: {query} (p={probability}, "
                f"adjusted_threshold={threshold})"
            )
            await weak_model_queue.put(query)

        return threshold

    async def process_queue(actor, queue, result_queue, model_name):
        while True:
            query = await queue.get()
            if query is None:
                break

            start_time = time.time()
            logger.info(
                f"{model_name} started processing query '{query}' at {datetime.now()}"
            )
            result = await send_request_to_model(actor, query)
            processing_time = time.time() - start_time
            logger.info(
                f"{model_name} finished processing query '{query}' at "
                f"{datetime.now()}, took {processing_time:.2f}s: {result}"
            )

            await asyncio.to_thread(
                save,
                {
                    "model_name": model_name,
                    "processing_time": processing_time,
                    "start_time": datetime.fromtimestamp(
                        start_time,
                        tz=timezone.utc,
                    ).isoformat(),
                    "query": query,
                    "result": result[0].outputs[0].text,
                },
            )

            await result_queue.put((query, result))

            queue.task_done()

    async def retrieve_results(result_queue, model_name):
        while True:
            query, result = await result_queue.get()

            if query is None:
                break

            logger.info(f"Result from {model_name}: {query} -> {result}")
            result_queue.task_done()

    async def monitor_queues(strong_model_queue, weak_model_queue):
        while True:
            logger.info(f"Strong Model Queue Depth: {strong_model_queue.qsize()}")
            logger.info(f"Weak Model Queue Depth: {weak_model_queue.qsize()}")
            await asyncio.sleep(1)

    async def run(queries):
        strong_queue = asyncio.Queue()
        weak_queue = asyncio.Queue()

        strong_results = asyncio.Queue()
        weak_results = asyncio.Queue()

        strong_task = asyncio.create_task(
            process_queue(
                strong_model,
                strong_queue,
                strong_results,
                "Strong Model",
            )
        )

        weak_task = asyncio.create_task(
            process_queue(
                weak_model,
                weak_queue,
                weak_results,
                "Weak Model",
            )
        )

        strong_result_task = asyncio.create_task(
            retrieve_results(
                strong_results,
                "Strong Model",
            )
        )

        weak_result_task = asyncio.create_task(
            retrieve_results(
                weak_results,
                "Weak Model",
            )
        )

        # Monitor queue depths
        monitor_task = asyncio.create_task(monitor_queues(strong_queue, weak_queue))

        threshold = 0.5
        learning_rate = 0.05

        # Simulate query arrival and route and process incoming queries
        for query in queries:
            threshold = await process_query(
                query,
                strong_queue,
                weak_queue,
                threshold,
                learning_rate,
                TARGET_RATIO,
            )
            await asyncio.sleep(1)

        # Signal completion
        await strong_queue.put(None)
        await weak_queue.put(None)

        # Wait for both models to finish
        await asyncio.gather(strong_task, weak_task)

        # Signal completion for result retrieval
        await strong_results.put((None, None))
        await weak_results.put((None, None))

        # Wait for result retrieval tasks to finish
        await asyncio.gather(strong_result_task, weak_result_task)

        # Cancel the monitoring task
        monitor_task.cancel()

    # Load dataset
    dataset = (
        load_dataset("sentence-transformers/eli5", split="train", streaming=True)
        .shuffle(SEED)
        .take(1000)
    )
    queries = [record["question"] for record in dataset]

    await run(queries)

    logger.info(f"Ended job {JOB_ID} at {datetime.now().isoformat()}")
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
