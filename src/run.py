import json
import os

## set API keys
os.environ['OPENAI_API_KEY'] = os.getenv('TRACTOR_OPENAI_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('TRACTOR_ANTHROPIC_API_KEY')

import sys
import csv
from pathlib import Path
import argparse
from benchmark import load_benchmarks
from prompter import (
    Prompter,
    RepairPrompter,
)
from transpiler import Transpiler, TranspilerN
from test_repairer import TestRepairer
from utils.rust_project_builder import write_rust_files, write_rust_multi_files, create_top_level_cargo_toml
from utils.compile_rust_utils import (
    final_error_report,
    get_errors_for_iteration,
    final_test_report,
    aggregate_results,
    performance_stats
)
import shutil
from endpoint_config import endpoint_resolver
from compile_projects import compile, test
from repairer import Repairer
from utils.parse_c import order_dependencies
from understand_errors import (
    get_numbers,
    process_proj
)


class Runner:
    def __init__(
        self,
        benchmark_dir,
        output_dir,
        prompt,
        prompt_format,
        prompt_strategy,
        repairer_prompt,
        repairer_format,
        repairer_strategy,
        iterations,
        endpoint,
        include_headers,
        config,
        n,
        single_benchmark,
        rust_dir,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)

        self.benchmarks = load_benchmarks(self.benchmark_dir, self.output_dir)
        self.prompt = prompt
        self.prompt_format = prompt_format
        self.prompt_strategy = prompt_strategy
        self.repairer_prompt = repairer_prompt
        self.repairer_format = repairer_format
        self.repairer_strategy = repairer_strategy
        self.iterations = int(iterations)
        self.endpoint = endpoint
        self.include_headers = include_headers
        self.config = config
        self.n = n
        if rust_dir:
            for benchmark in self.benchmarks:
                shutil.rmtree(benchmark.rust_path)
                if not Path(rust_dir + "/" + benchmark.project_name).exists():
                    # try capitalizing the first letter of the project name
                    shutil.copytree(
                        rust_dir + "/" + benchmark.project_name[0].upper() + benchmark.project_name[1:],
                        benchmark.rust_path,
                    )
                else:
                    shutil.copytree(
                        rust_dir + "/" + benchmark.project_name, benchmark.rust_path
                    )
            # we need to reload the benchmarks given that we have changed the rust path
            self.benchmarks = load_benchmarks(self.benchmark_dir, self.output_dir)
            assert all(
                len(benchmark.rust_headers) > 0 for benchmark in self.benchmarks
            ), "Rust path not copied correctly"
        if single_benchmark:
            # check if start with a number
            if single_benchmark[0].isdigit():
                single_benchmark = "proj_" + single_benchmark
            single_benchmark = single_benchmark.replace("-", "_")
            self.benchmarks = [
                b for b in self.benchmarks if b.project_name == single_benchmark
            ]
        # dump a config log
        print(f'''
        Number of benchmarks: {len(self.benchmarks)}
        Prompt: {self.prompt}
        Prompt format: {self.prompt_format}
        Prompt strategy: {self.prompt_strategy}
        Repairer prompt: {self.repairer_prompt}
        Repairer format: {self.repairer_format}
        Repairer strategy: {self.repairer_strategy}
        Iterations: {self.iterations}
        Endpoint: {self.endpoint}
        Include interfaces: {self.include_headers}
        Config: {self.config}
        ''')
 

    def transpile(self):
        prompter = Prompter(
            self.prompt, self.prompt_format, self.prompt_strategy, self.include_headers
        )

        transpiler = Transpiler(
            self.benchmarks, prompter, endpoint=self.endpoint, config=self.config
        )
        results, benchmarks = transpiler.run()
        self.benchmarks = benchmarks
        transpiler.log_benchmarks(results)
        for benchmark in self.benchmarks:
            write_rust_files(benchmark)
        compile_results = compile(self.output_dir, 0)

    def repair(self):
        compile_results = []
        for i in range(self.iterations):
            repair_prompter = RepairPrompter(
                self.repairer_prompt, self.repairer_format, self.repairer_strategy
            )
            repairer = Repairer(
                self.benchmarks, repair_prompter, i, self.endpoint, config=self.config
            )
            results, benchmarks = repairer.run()
            self.benchmarks = benchmarks
            repairer.log_results(results)
            for benchmark in self.benchmarks:
                write_rust_files(benchmark)
            compile_results.append(compile(self.output_dir, i + 1))

    
        

    def multi_gen(self):
        prompter = Prompter(
            self.prompt, self.prompt_format, self.prompt_strategy, self.include_headers
        )

        transpiler = TranspilerN(
            self.benchmarks, prompter, endpoint=self.endpoint, config=self.config, n=self.n
        )
        results, benchmarks_list = transpiler.run()
        for idx, benchmarks in enumerate(benchmarks_list):
            for benchmark in benchmarks:
                write_rust_multi_files(benchmark, idx)
            # create top level directory for each iteration
            create_top_level_cargo_toml(benchmarks[0][0].rust_path.parent / f"{idx}")
        # clean up directories that are not used
        for dir in self.output_dir.iterdir():
            # check if dir name is a number
            if dir.is_dir() and not dir.name.isdigit():
                shutil.rmtree(dir)
        
        # get ready for repair
        self.benchmarks = []
        for i in range(self.n):
            self.benchmarks = load_benchmarks(self.benchmark_dir , self.output_dir / str(i))
            compile_results = compile(self.output_dir / str(i), 0)
            self.repair()
            self.get_loc()
            final_error_report(self.output_dir / str(i))
            final_test_report(self.output_dir / str(i))
            get_errors_for_iteration(self.output_dir / str(i))
            performance_stats(self.output_dir / str(i))
            self.benchmarks = []
        # aggregate the results based on the benchmark name
        aggregate_results([self.output_dir / str(i) for i in range(self.n)], self.output_dir)

    def get_loc(self):
        with open(self.output_dir / "loc.csv", "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["project", "c_loc", "rust_loc", "builds"])
            for benchmark in self.benchmarks:
                res = benchmark.compile_rust()
                c_loc, rust_loc = benchmark.count_loc()
                if res == []:
                    if rust_loc == 3:
                        builds = "Empty Project"
                    else:
                        builds = "True"
                else:
                    builds = "False"

                csv_writer.writerow([benchmark.project_name, c_loc, rust_loc, builds])

    def test_perf(self):
        self.transpile()
        self.repair()
        self.get_loc()
        # errors per iteration of repair
        get_errors_for_iteration(self.output_dir)
        # errors after all iterations have been run
        final_error_report(self.output_dir)
        # test based errors after all iterations have been run
        final_test_report(self.output_dir)
        # performance stats per project
        performance_stats(self.output_dir)
        # categorize errors based on error type
        process_proj(self.output_dir)
        # get the final numbers
        get_numbers(self.output_dir)


def main(args):
    config = endpoint_resolver(args.config, args.endpoint)
    runner = Runner(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        prompt_format=args.prompt_format,
        prompt_strategy=args.prompt_strategy,
        repairer_prompt=args.repairer_prompt,
        repairer_format=args.repairer_format,
        repairer_strategy=args.repairer_strategy,
        iterations=args.iterations,
        endpoint=args.endpoint,
        include_headers=args.include_headers,
        config=config,
        single_benchmark=args.single_benchmark,
        rust_dir=args.rust_dir,
        n=args.n,
    )
    config = {}
    print("mode:", args.mode)
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    if args.mode == "normal":
        runner.test_perf()
    elif args.mode == "multi_gen":
        print(f"Top-{args.n} generation, with temperature {config['temperature']}")
        runner.multi_gen()
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Evaluate your model with CRUST-bench")
    argparser.add_argument("--benchmark_dir", type=str, required=True, help="Path to the C project (CBench) directory")
    argparser.add_argument("--rust_dir", type=str, required=False, help="Path to the Rust project (RBench) directory")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    argparser.add_argument("--prompt", type=str, required=True, help="Prompt to use for the model")
    argparser.add_argument("--mode", type=str, default="normal", help="The mode(normal, multi_gen), that does the transpilation")
    argparser.add_argument("--endpoint", type=str, required=True, help="Endpoint to use for the model. Look at the `endpoints/call_endpoint.py` for more information.")
    argparser.add_argument("--prompt_format", type=str, required=True, help="Format of the prompt (markdown, bullet_point)")
    argparser.add_argument("--prompt_strategy", type=str, required=False, default="all", help="Strategy to use for the prompt (all- all files are appended to the prompt)")
    argparser.add_argument("--repairer_prompt", type=str, required=True, help="Prompt to use for the repairer")
    argparser.add_argument("--repairer_format", type=str, required=True, help="Format of the repairer prompt(markdown, bullet_point)")
    argparser.add_argument("--repairer_strategy", type=str, required=True, help="Strategy to use for the repairer prompt(all- all files are appended to the prompt)")
    argparser.add_argument("--iterations", type=str, required=True, help="Number of iterations to run the repairer")
    argparser.add_argument("--include_headers", type=bool, default=True, help="Whether to include headers in the prompt")
    argparser.add_argument("--single_benchmark", type=str, default=None, help="Set this flag when you only want to run a single benchmark to run")
    argparser.add_argument("--config", type=str, required=False, default=None, help="Path to the endpoint config file")
    argparser.add_argument("--n", type=int, default=1, help="Number of generations to receive from the model during transpilation")
    args = argparser.parse_args()
    main(args)
