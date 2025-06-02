import subprocess
import os
from tqdm import tqdm
from pathlib import Path
import shutil

DATASETS_PATH = Path(__file__).parent / '../../datasets/'

def compile_rust_test_proj(proj_name):
    env = os.environ.copy()
    env["RUST_FLAGS"] = "-Awarnings"
    try:
        result = subprocess.run(
            ["cargo", "build"], cwd=proj_name, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for {proj_name}")
        return False
    std_out = result.stdout
    std_err = result.stderr
    if 'error:' in std_err:
        return False
    return True

def clean_rust_project(proj_name):
    env = os.environ.copy()
    env["RUST_FLAGS"] = "-Awarnings"
    r = subprocess.run(
        ["cargo", "clean"], cwd=proj_name, capture_output=True, text=True, env=env
    )
    return True


def compile_rust_projs():
    path = DATASETS_PATH / 'RBench_formatted'
    ctr = 0
    error_list = []
    proj_list = list(path.iterdir())
    for proj in tqdm(proj_list):
        if proj.is_dir():
            flg= compile_rust_test_proj(proj)
            if not flg:
                ctr+=1
                error_list.append(proj)
    print(ctr)
    with open('error_list.txt', 'w') as f:
        for item in error_list:
            f.write("%s\n" % item)
    print('done')

def format_into_compilable_rust(proj_name):
    src_path = proj_name / 'src'
    interfaces_path = proj_name / 'src' / 'interfaces'
    # move interface files to src
    for file in interfaces_path.iterdir():
        if file.is_file() and file.suffix == '.rs':
            shutil.move(file, src_path)
    # remove the interfaces folder
    if interfaces_path.exists():
        shutil.rmtree(interfaces_path)

def format_rust_proj():
    path = DATASETS_PATH / 'RBench'
    output_path = DATASETS_PATH / 'RBench_formatted'
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    proj_list = list(path.iterdir())
    for proj in tqdm(proj_list):
        if not proj.is_dir():
            continue
        shutil.copytree(proj, output_path / proj.name, dirs_exist_ok=True)
        format_into_compilable_rust(output_path / proj.name)

        
def clean_rust_projects():
    path = DATASETS_PATH / 'RBench_formatted'
    proj_list = list(path.iterdir())
    for proj in tqdm(proj_list):
        if proj.is_dir():
            clean_rust_project(proj)
    print('done')

if __name__ == "__main__":    
    format_rust_proj()
    compile_rust_projs()
    clean_rust_projects()