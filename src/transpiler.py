import multiprocessing
from multiprocessing import Pool
import json
from pathlib import Path
from endpoints.call_endpoint import get_result, get_result_n
from utils.compile_rust_utils import compile_all_rust_proj
from benchmark import Benchmark 
from tqdm import tqdm
from utils.parse_c import order_dependencies
from functools import partial
import copy
from utils.compute_costs import compute_cost_of_prompts
import time

def get_results_benchmark(lock_and_prompts, endpoint, config):
    lock, prompts = lock_and_prompts
    result = []
    for prompt in prompts[1]:
        result.append(
            (prompts[0], get_result(prompt, lock, endpoint, config)["response"])
        )
        if endpoint == 'claude':
            time.sleep(20)
    return result

def get_results_benchmark_n(lock_and_prompts, endpoint, config, n=5):
    lock, prompts = lock_and_prompts
    result = []
    for prompt in prompts[1]:
        result.append(
            (prompts[0], get_result_n(prompt, lock, endpoint, config, n)["response"])
        )
    return result

class Transpiler:
    def __init__(self, benchmarks, prompter, endpoint, config):
        self.benchmarks = benchmarks
        self.prompter = prompter
        self.manager = multiprocessing.Manager()
        self.lock = self.manager.Lock()
        self.endpoint = endpoint
        self.benchmark_name_to_benchmark = {}
        self.config = config

    def log_benchmarks(self, results):
        for result in results:
            for ben_res in result:
                (cur_bench, response) = ben_res
                metadata_path = cur_bench.rust_path / "metadata"
                output_metadata_path = metadata_path / "output"
                if not output_metadata_path.exists():
                    output_metadata_path.mkdir(parents=True)
                with open(
                    output_metadata_path / "initial.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(response)

    def dump_transpilation_prompt(self, prompt, benchmark):
        metadata_path = benchmark.rust_path / "metadata"
        output_metadata_path = metadata_path / "output" / "prompt" / "transpilation"
        if not output_metadata_path.exists():
            output_metadata_path.mkdir(parents=True)
        with open(
            output_metadata_path / "transpilation_prompt.json", "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(prompt))

    def run(self):
        prompts = []
        benchmarks = []
        for benchmark in tqdm(self.benchmarks):
            prompt = self.prompter(benchmark)
            prompts.append((benchmark, prompt))
            self.dump_transpilation_prompt(prompt, benchmark)
        print("Computing costs")
        if self.config["model"] in ["o1", "o1-mini", "gpt-4o", "claude-3-5-sonnet-20241022", "gemini", "claude-3-7-sonnet-20250219"]:
            compute_cost_of_prompts(prompts, self.config["model"])
            while True:
                user_input = input("Continue? (y/n): ")
                if user_input.lower() in ["y", "n"]:
                    break
            if user_input.lower() == "n":
                exit(0)
        get_result_fn = partial(
            get_results_benchmark, endpoint=self.endpoint, config=self.config
        )
        results = []
        if "parallel" in self.config and self.config["parallel"] in ["False", False]:
            for prompt in prompts:
                results.append(get_result_fn((self.lock, prompt)))
        else:
            processes = 8
            if self.config["model"] == "claude-3-5-sonnet-20241022" or self.config["model"] == "claude-3-7-sonnet-20250219":
                processes = 1
            with Pool(processes=processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(
                            get_result_fn,
                            [(self.lock, prompt) for prompt in prompts],
                        ),
                        total=len(prompts),
                    )
                )
        
        for result in results:
            for ben_res in result:
                (curr_benchmark, response) = ben_res
                parsed = self.prompter.parse_response(response)
                current_benchmark_rust_files = {
                    p["file_name"]: p["content"] for p in curr_benchmark.rust_files
                }
                for p in parsed:
                    current_benchmark_rust_files[p["file_name"]] = p["content"]
                curr_benchmark.rust_files = [
                    {"file_name": k, "content": v}
                    for k, v in current_benchmark_rust_files.items()
                ]
                benchmarks.append(curr_benchmark)
        return results, benchmarks

class TranspilerN:
    def __init__(self, benchmarks, prompter, endpoint, config, n):
        self.benchmarks = benchmarks
        self.prompter = prompter
        self.manager = multiprocessing.Manager()
        self.lock = self.manager.Lock()
        self.endpoint = endpoint
        self.benchmark_name_to_benchmark = {}
        self.config = config
        self.n = n
    def log_benchmarks(self, results):
        pass 
    def dump_transpilation_prompt(self, prompt, benchmark):
        metadata_path = benchmark.rust_path / "metadata"
        output_metadata_path = metadata_path / "output" / "prompt" / "transpilation"
        if not output_metadata_path.exists():
            output_metadata_path.mkdir(parents=True)
        with open(
            output_metadata_path / "transpilation_prompt.json", "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(prompt))
    def run(self):
        prompts = []
        benchmarks = []
        for benchmark in tqdm(self.benchmarks):
            prompt = self.prompter(benchmark)
            prompts.append((benchmark, prompt))
            self.dump_transpilation_prompt(prompt, benchmark)
        print("Computing costs")
        if self.config["model"] in ["o1", "o1-mini", "gpt-4o", "claude", "gemini"]:
            total_tokens, total_cost = compute_cost_of_prompts(prompts, self.config["model"], self.n)
            with open(self.benchmarks[0].rust_path.parent / "costs.txt", "w") as f:
                f.write(f"Total tokens: {total_tokens}\n")
                f.write(f"Total cost: {total_cost}\n")
                f.write(f"Model: {self.config['model']}\n")
                f.write(f"NOTE: This is for a single pass")
            while True:
                user_input = input("Continue? (y/n): ")
                if user_input.lower() in ["y", "n"]:
                    break
            if user_input.lower() == "n":
                exit(0)
        get_result_fn = partial(
            get_results_benchmark_n, endpoint=self.endpoint, config=self.config, n=self.n
        )
        with Pool(processes=24) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        get_result_fn,
                        [(self.lock, prompt) for prompt in prompts],
                    ),
                    total=len(prompts),
                )
            )
        benchmark_n = [[] for _ in range(self.n)]
        for result in results:
            for ben_res in result:
                (curr_benchmark, response) = ben_res
                for idx, candidate in enumerate(response):
                    parsed = self.prompter.parse_response(candidate)
                    b_cpy = copy.deepcopy(curr_benchmark)
                    for p in parsed:
                        b_cpy.rust_files.append(
                            {"file_name": p["file_name"], "content": p["content"]}
                        )
                    benchmark_n[idx].append((b_cpy, candidate))
        return results, benchmark_n

if __name__ == "__main__":
    pass