# CRUST-Bench: A Comprehensive Benchmark for C-to-safe-Rust Transpilation

Authors: Anirudh Khatry, Robert Zhang, Jia Pan, Ziteng Wang, Qiaochu Chen, Greg Durrett, Isil Dillig.

![Dataset Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-GNU%20GPL-blue)
![Last Updated](https://img.shields.io/badge/last%20updated-April%202025-orange)

![CRUST-bench workflow](./src/resources/CRUST-bench.png)

## Overview

C-to-Rust transpilation is essential for modernizing legacy C code while enhancing safety and interoperability with modern Rust ecosystems. However, no dataset currently exists for evaluating whether a system can transpile C into safe Rust that passes a set of test cases. We introduce CRUST-Bench, a dataset of 100 C repositories, each paired with manually-written interfaces in safe Rust as well as test cases that can be used to validate correctness of the transpilation. By considering entire repositories rather than isolated functions, CRUST-Bench captures the challenges of translating complex projects with dependencies across multiple files. The provided Rust interfaces provide explicit specifications that ensure adherence to idiomatic, memory-safe Rust patterns, while the accompanying test cases enforce functional correctness. We evaluate state-of-the-art large language models (LLMs) on this task and find that safe and idiomatic Rust generation is still a challenging problem for various state-of-the-art methods and techniques. We also provide insights into the errors LLMs usually make in transpiling code from C to safe Rust. The best performing model, OpenAI o1, is able to solve only 15 tasks in a single-shot setting. Improvements on CRUST-Bench would lead to improved transpilation systems that can reason about complex scenarios and help in migrating legacy codebases from C into languages like Rust that ensure memory safety.

## Paper

Our paper "CRUST-Bench: A Comprehensive Benchmark for C-to-safe-Rust Transpilation" is available at:
- [arXiv](https://arxiv.org/abs/2504.15254)

## Dataset Description

### Data Collection
We collected data from 100 Github repositories, spanning various domains like:
 - System utilities
 - Algorithms
 - Programming Language Infrastructure
 - Networking
 - Cryptography and Security
 - Data structures 
 - etc.


### Data Format
The dataset consists of 2 folders namely:
1. CBench
2. Rust Bench

```
CBench/
├── Project_1/
│   ├── file1.c/
│   ├── file2.c/
│   └── ...
├── Project_2/
│   ├── file1.c/
│   ├── file2.c/
│   └── ...
├── Project_3/
│   ├── file1.c/
│   ├── file2.c/
│   └── ...
└── ...

RBench/
├── Project_1/
│   ├── interfaces
│   │   ├── file1.rs
│   │   ├── file2.rs
│   │   └── ...
│   ├── bin
│   │   ├── test1.rs
│   │   ├── test2.rs
│   │   └── ...
├── Project_2/
│   ├── interfaces
│   │   ├── file1.rs
│   │   ├── file2.rs
│   │   └── ...
│   ├── bin
│   │   ├── test1.rs
│   │   ├── test2.rs
│   │   └── ...
│   └── ...
└── ...


```

## Usage

### Requirements
List any dependencies required to use the dataset:

```bash
pip install -r requirements.txt
```

### Environment Setup

Rust is required to build the projects. You can install Rust using [rustup](https://www.rust-lang.org/tools/install).

### Loading the Dataset
The dataset is within the `datasets` folder as a zip file that can be extracted to provide 2 folders:
  1. __CBench__: the projects scraped from github.
  2. __RBench__: the manually annotated interfaces and corresponding tests.

NOTE: You must ensure that both `CBench` and `RBench` folders are in the `datasets/`

To perform a sanity type check, we provide the `check_benchmarks/check_build.py` script that produces a compilable version of the rust project with the `unimplemented!()` function bodies that can be type checked. To run the script, you need to have `rust` installed on you system. You can run the script as follows:

```python
python src/check_benchmarks/check_build.py
```

The `src/dataset_stats` aids in plotting metrics over the C repositories and the annotated Rust interfaces and tests.

To get the statistics of the dataset, you can run the following command:

```python
python src/dataset_stats/get_c_stats.py datasets/CBench
python src/dataset_stats/get_interfaces_stats.py datasets/RBench
```

### Recreating experiments from CRUST-bench.
Please set the relevant OpenAI, Antropic, Google CLOUD API keys using the environment variables.

```bash
export OPENAI_API_KEY=<OpenAI_API_KEY>
export ANTHROPIC_API_KEY=<ANTHROPIC_API_KEY>
export GOOGLE_CLOUD_PROJECT=<GOOGLE_CLOUD_PROJECT>
export GOOGLE_CLOUD_REGION=<GOOGLE_CLOUD_REGION>
```

We provide easy bash scripts located in the `scripts` folder that allow you to easily test your models with our pipeline. 

The entrypoint to our code is the `src/run.py` file that takes in the following params:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--benchmark_dir` | `str` | ✅ | – | Path to the C project (**CBench**) directory. |
| `--rust_dir` | `str` | ❌ | `None` | Path to the Rust project (**RBench**) directory. |
| `--output_dir` | `str` | ✅ | – | Path to the output directory. |
| `--prompt` | `str` | ✅ | – | Prompt to use for the model during transpilation. |
| `--mode` | `str` | ❌ | `"normal"` | Transpilation mode. Options: `normal`, `multi_gen`. |
| `--endpoint` | `str` | ✅ | – | Endpoint to use for the model. See `endpoints/call_endpoint.py` for details. |
| `--prompt_format` | `str` | ✅ | – | Format of the prompt. Options: `markdown`, `bullet_point`. |
| `--prompt_strategy` | `str` | ❌ | `"all"` | Strategy for composing the prompt. Options: `all` (all files are appended). |
| `--repairer_prompt` | `str` | ✅ | – | Prompt used for the repairer model. |
| `--repairer_format` | `str` | ✅ | – | Format of the repairer prompt. Options: `markdown`, `bullet_point`. |
| `--repairer_strategy` | `str` | ✅ | – | Strategy for repairer prompt. Options: `all` (all files are appended). |
| `--iterations` | `str` | ✅ | – | Number of iterations to run the repairer. |
| `--include_headers` | `bool` | ❌ | `True` | Whether to include header files in the prompt. |
| `--single_benchmark` | `str` | ❌ | `None` | Run a single benchmark only (provide its name). |
| `--config` | `str` | ❌ | `None` | Path to the endpoint configuration file. |
| `--n` | `int` | ❌ | `1` | Number of generations to request from the model during transpilation. |

For instance to run with OpenAI's o1 model you can run the following command:

```bash
python run.py \
    --benchmark_dir "../datasets/CBench" \
    --output_dir "outputs/o1" \
    --prompt ./prompts/transpilation_prompts/bullet_point/bullet_point_interface.prompt \
    --prompt_format bullet_point_with_system_instructions \
    --prompt_strategy all \
    --repairer_prompt ./prompts/repair_prompts/bullet_point/bullet_point.prompt \
    --repairer_format bullet_point_with_system_instructions \
    --repairer_strategy all \
    --iterations 3 \
    --mode ./endpoints/configs/o1.json \
    --endpoint "o1" \
    --rust_dir "../datasets/RBench"
```

For running more scripts, please refer to the `scripts` folder.

In order to perform the test based repair:

First we need to run the default transpile-then-repair pipeline. You must rust the code above to get the initial Rust projects. The output of the above command is stored in the `outputs/o1` directory.

Next, we need to run the test based repair. This can be done by running the following command:
```bash
python repair_tests.py \
    --input_path outputs/o1 \
    --output_path output/o1_test_guided_repair \
    --endpoint o1 \
    --iterations 3
```
This will select the projects that compile but fail the tests and run the repairer on them. The repaired code will be saved in the `output/o1_test_guided_repair` directory.

The final result will be saved in the `output/o1_test_guided_repair` directory.

### Adding your own models
To add your own model, you must:
1. Define the model in the `endpoints` folder. You must ensure that your model implements a `get_results` function that takes in a message and returns the response from the model in the form of a json that contains the `response` key.
2. Define the model configuration in the `configs` folder. The configuration file should contain the model name, the endpoint to call, and any other relevant parameters.
3. Update the `call_endpoint.py` file to include the new model in the `endpoints` dictionary. The `endpoints` dictionary should map the model name to the corresponding endpoint function.

## Citation

If you use this dataset in your research, please cite our paper:

```bibtex
@misc{
  khatry2025crustbenchcomprehensivebenchmarkctosaferust,
  title={CRUST-Bench: A Comprehensive Benchmark for C-to-safe-Rust Transpilation}, 
  author={Anirudh Khatry and Robert Zhang and Jia Pan and Ziteng Wang and Qiaochu Chen and Greg Durrett and Isil Dillig},
  year={2025},
  eprint={2504.15254},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2504.15254}, 
}
```



## License

This dataset is released under GNU GPL license. See [LICENSE](LICENSE) for details.

## Contact

For questions, issues, or further information, please contact:
- **Name**: Anirudh Khatry
- **Email**: [akhatry@utexas.edu]
- **GitHub**: [@anirudhkhatry](https://github.com/anirudhkhatry)

## Acknowledgments

This research was conducted within a group supported by the National Science Foundation
under awards CCF-1762299, CCF-1918889, CNS-1908304, CCF-1901376, CNS-2120696, CCF-
2210831, and CCF-2319471. We also thank the All Hands AI team for a discussion on the
OpenHands CodeAct agentic framework applied to the C-to-Rust transpilation task.