import json
import os
from pathlib import Path
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import glob

# Reusing your existing parser setup
from tree_sitter import Language, Parser

FILE_PATH = Path(__file__)

Language.build_library(
    str(FILE_PATH.parent / "c_build/my-languages.so"),
    [FILE_PATH.parent.parent / "resources/tree-sitter-c"],
)

C_LANGUAGE = Language(str(FILE_PATH.parent / "c_build/my-languages.so"), "c")
PARSER = Parser()
PARSER.set_language(C_LANGUAGE)

class CProject:
    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.c_files = []
        self.h_files = []
        self.test_files = []
        self.source_files = []
        self.load_files()
        
    def load_files(self):
        # Find all C and H files, excluding test directories
        for file_path in self.project_path.glob("**/*.c"):
            # print(file_path)
            # Skip files in test directories
            if "test" in str(file_path).lower() or "bin" in str(file_path.parent).lower() or "bin" in str(file_path.parent.parent).lower():
                self.test_files.append(file_path)
                continue
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                self.c_files.append({
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "content": content
                })
                
        for file_path in self.project_path.glob("**/*.h"):
            # print(file_path)
            # Skip files in test directories
            if "test" in str(file_path).lower().split(os.path.sep):
                self.test_files.append(file_path)
                continue
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                self.h_files.append({
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "content": content
                })
        self.source_files = self.c_files + self.h_files

    def count_lines_of_code(self):
        """Count non-empty, non-comment lines of code in C files"""
        total_lines = 0
        for file in self.source_files:
            content = file["content"]
            # Remove C-style comments
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            # Remove C++-style comments
            content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
            # Count non-empty lines
            lines = [line for line in content.split('\n') if line.strip()]
            total_lines += len(lines)
        return total_lines

    def count_functions(self):
        """Count function definitions in C files"""
        function_count = 0
        for file in self.source_files:
            content = file["content"]
            tree = PARSER.parse(content.encode("utf8"))
            root_node = tree.root_node
            
            # Count function definitions
            function_nodes = []
            
            def traverse(node):
                if node.type == "function_definition":
                    function_nodes.append(node)
                for child in node.children:
                    traverse(child)
            
            traverse(root_node)
            function_count += len(function_nodes)
            
        return function_count

    def count_pointer_references(self):
        """Count pointer references (*) in C files"""
        pointer_count = 0
        for file in self.source_files:
            content = file["content"]
            tree = PARSER.parse(content.encode("utf8"))
            root_node = tree.root_node
            
            # Count pointer declarators and dereference expressions
            pointer_nodes = []
            
            def traverse(node):
                if node.type == "pointer_declarator" or node.type == "pointer_expression":
                    pointer_nodes.append(node)
                for child in node.children:
                    traverse(child)
            
            traverse(root_node)
            pointer_count += len(pointer_nodes)
            
        return pointer_count

    def count_test_cases(self):
        """
        Count test cases by looking for test files and typical test function patterns
        """
        test_count = 0
        test_patterns = [
            r'test_\w+\s*\(',  # Functions starting with test_
            r'TEST\s*\(',      # TEST macro
            r'TEST_F\s*\(',    # TEST_F macro (googletest)
            r'BOOST_TEST',     # Boost test macros
            r'CHECK\s*\(',     # Various CHECK macros
            r'ASSERT\s*\(',    # ASSERT macros
            r'assert\s*\(',    # assert function
            r'skptest\s*\(',   # skia test function
        ]
        
        # First check test files we identified during loading
        for file_path in self.test_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for pattern in test_patterns:
                        test_count += len(re.findall(pattern, content))
            except:
                pass
        
        # Then also check any "test" directories we might have missed
        for test_dir in glob.glob(f"{self.project_path}/**/test*", recursive=True):
            if os.path.isdir(test_dir):
                for file_path in glob.glob(f"{test_dir}/**/*.c", recursive=True):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            for pattern in test_patterns:
                                test_count += len(re.findall(pattern, content))
                    except:
                        pass
        
        return test_count

    def attempt_get_code_coverage(self):
        """
        Try to get code coverage from coverage files
        Returns None if no coverage data is available
        """
        # Look for coverage files (lcov info or gcov data)
        coverage_files = list(self.project_path.glob("**/*.info")) + \
                         list(self.project_path.glob("**/*.gcov"))
        
        if coverage_files:
            # Parse actual coverage data if available
            total_lines = 0
            covered_lines = 0
            
            for cov_file in coverage_files:
                try:
                    with open(cov_file, 'r') as f:
                        content = f.read()
                        if cov_file.suffix == '.info':  # lcov format
                            # Simple parsing of lcov format
                            covered = re.findall(r'LH:(\d+)', content)
                            total = re.findall(r'LF:(\d+)', content)
                            if covered and total:
                                covered_lines += int(covered[0])
                                total_lines += int(total[0])
                        elif cov_file.suffix == '.gcov':  # gcov format
                            # Count executed lines vs. total executable lines
                            lines = content.split('\n')
                            for line in lines:
                                if re.match(r'^\s*\d+:', line):  # Executed line
                                    covered_lines += 1
                                    total_lines += 1
                                elif re.match(r'^\s*#####:', line):  # Not executed
                                    total_lines += 1
                except:
                    pass
            
            if total_lines > 0:
                return (covered_lines / total_lines) * 100
        
        # If we get here, no coverage data was found or parsed
        return None
    
    def get_project_stats(self):
        """
        Generate a dictionary with all the project statistics
        """
        stats = {
            "c_files": len(self.c_files),
            "h_files": len(self.h_files),
            "lines_of_code": self.count_lines_of_code(),
            "functions": self.count_functions(),
            "pointer_references": self.count_pointer_references(),
            "test_cases": self.count_test_cases(),
            "test_files": len(self.test_files),
        }
        
        # Try to get code coverage, but it might not be available
        coverage = self.attempt_get_code_coverage()
        if coverage is not None:
            stats["code_coverage"] = coverage
        
        return stats

def analyze_c_projects(project_paths):
    """
    Analyze multiple C projects and return stats as a dictionary
    """
    results = {}
    
    for path in project_paths:
        proj_name = os.path.basename(os.path.normpath(path))
        try:
            print(f"Processing {proj_name}...")
            project = CProject(path)
            results[proj_name] = project.get_project_stats()
            print(f"  Found {len(project.c_files)} C files")
            print(f"  Found {len(project.h_files)} H files")
            print(f"  Found {results[proj_name]['lines_of_code']} lines of code")
            print(f"  Found {results[proj_name]['functions']} functions")
            print(f"  Found {results[proj_name]['pointer_references']} pointer references")
            print(f"  Found {results[proj_name]['test_cases']} test cases")
            
            
            if "code_coverage" in results[proj_name]:
                print(f"  Code coverage: {results[proj_name]['code_coverage']:.2f}%")
            else:
                print("  No code coverage data available")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            results[proj_name] = {"error": str(e)}
    
    return results

def main():
    """
    Process multiple C projects and save their statistics as JSON
    """
    if len(sys.argv) < 2:
        print("Usage: python c_stats_to_json.py project1_path [project2_path ...]")
        sys.exit(1)
    
    project_path = Path(sys.argv[1])
    project_paths = []
    for path in project_path.iterdir():
        if path.is_dir():
            project_paths.append(path)
    results = analyze_c_projects(project_paths)
    
    output_file = "c_project_stats.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nStatistics saved to {output_file}")

if __name__ == "__main__":
    main()