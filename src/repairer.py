import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
from endpoints.call_endpoint import get_result
import json
from functools import partial
import time

def get_results_benchmark(lock_and_prompts, endpoint, config):
    lock, prompts = lock_and_prompts
    result = []
    for prompt in prompts[1]:
        run_result = get_result(prompt, lock, endpoint, config)
        result.append(
            (prompts[0], run_result["response"], run_result["reasoning"])
        )
        if endpoint == 'claude':
            time.sleep(20)
    return result


class Repairer:
    def __init__(self, benchmarks, repair_prompter, iteration, endpoint, config):
        self.benchmarks = benchmarks
        self.repair_prompter = repair_prompter
        self.iteration = iteration
        self.manager = multiprocessing.Manager()
        self.lock = self.manager.Lock()
        self.endpoint = endpoint
        self.config = config

    def write_benchmarks(self, results):
        bechmarks = []
        for result in results:
            for ben_res in result:
                (cur_bench, response, _) = ben_res
                parsed = self.repair_prompter.parse_response(response)
                rust_files = cur_bench.rust_files.copy()
                cur_bench.rust_files = []
                all_files = set([f["file_name"] for f in rust_files]).union(
                    set([f["file_name"] for f in parsed])
                )
                for file in all_files:
                    found = False
                    for f in parsed:
                        if f["file_name"] == file:
                            cur_bench.rust_files.append(f)
                            found = True
                            break

                    # try adding files that weren't repaired
                    if not found:
                        for f in rust_files:
                            if f["file_name"] == file:
                                cur_bench.rust_files.append(f)
                                break
                bechmarks.append(cur_bench)
        return bechmarks

    def log_results(self, results):
        for result in results:
            for ben_res in result:
                (cur_bench, response, reasoning) = ben_res
                metadata_path = cur_bench.rust_path / "metadata"
                output_metadata_path = metadata_path / "output"
                if not output_metadata_path.exists():
                    output_metadata_path.mkdir(parents=True)
                with open(
                    output_metadata_path / f"res_{self.iteration}.txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(response)
                print(
                    f"Written to {output_metadata_path / f'res_{self.iteration}.txt'}"
                )
                with open(
                    output_metadata_path / f"res_{self.iteration}_reasoning.txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(reasoning)

    def run(self):
        prompts = []
        other_benchmarks = []
        for benchmark in self.benchmarks:
            if Path(
                benchmark.rust_path / "metadata" / f"errors_{self.iteration}.json"
            ).exists():
                
                prompt = self.repair_prompter(benchmark, self.iteration)

                prompts.append((benchmark, prompt))
            else:
                other_benchmarks.append(benchmark)
        get_result_fn = partial(
            get_results_benchmark, endpoint=self.endpoint, config=self.config
        )
        results = []
        if "parallel" in self.config and self.config["parallel"]=="False":
            for prompt in prompts:
                results.append(get_result_fn((self.lock, prompt)))
        else:
            with Pool(processes=8) as pool:
                results = list(
                    tqdm(
                        pool.imap(
                            get_result_fn,
                            [(self.lock, prompt) for prompt in prompts],
                        ),
                        total=len(prompts),
                    )
                )
        benchmarks = self.write_benchmarks(results)
        benchmarks.extend(other_benchmarks)
        return results, benchmarks


if __name__ == "__main__":
    pass
