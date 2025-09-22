from pathlib import Path
import json
import os
from tqdm import tqdm
from tree_sitter import Language, Parser
import sys
import re
from collections import defaultdict
# Use the existing tree-sitter setup
FILE_PATH = Path(__file__)

from utils.parse_rust import get_rust_functions

Language.build_library(
    str(FILE_PATH.parent / "rust_build/my-languages.so"),
    [FILE_PATH.parent.parent / "resources/tree-sitter-rust"],
)

RUST_LANGUAGE = Language(str(FILE_PATH.parent / "rust_build/my-languages.so"), "rust")
EMPTY_MAIN_STRING = 'fn main() {\n    println!("Hello, world!");\n}'
# Initialize the parser
PARSER = Parser()
PARSER.set_language(RUST_LANGUAGE)
class RustProjectAnalyzer:

    def __init__(self):
        self.project_stats = {}
        self.cumulative_stats = {
            "total_files": 0,
            "interface_files": 0,
            "total_functions": 0,
            "borrowed_arguments": 0,
            "custom_arg_types": set(),
            "custom_return_types": set(),
            "pointers": 0,
            "total_arguments": 0,
            "pointer_like_types": 0,
            "lifetime_annotations": 0,
            "mut_arguments": 0,
            "trait_bounds": 0,
            "custom_types": {
                "structs": set(),
                "enums": set(),
                "type_aliases": set(),
                "traits": set()
            },
            "traits_implemented": {
                "Clone": 0,
                "Debug": 0,
                "Copy": 0,
                "PartialEq": 0,
                "PartialOrd": 0,
                # Add more traits as needed
            }
        }
    
    def analyze_project(self, project_path):
        """Analyze a single Rust project."""
        project_name = os.path.basename(project_path)
        self.project_stats[project_name] = {
            "total_files": 0,
            "interface_files": 0,
            "functions": [],
            "borrowed_arguments": 0,
            "custom_arg_types": set(),
            "custom_return_types": set(),
            "pointers": 0,
            "mut_arguments": 0,
            "total_arguments": 0,
            "pointer_like_types": 0,
            "lifetime_annotations": 0,
            "trait_bounds": 0,
            
            "custom_types": {
                "structs": set(),
                "enums": set(),
                "type_aliases": set(),
                "traits": set()
            },
            "traits_implemented": {
                "Clone": 0,
                "Debug": 0,
                "Copy": 0,
                "PartialEq": 0,
                "PartialOrd": 0,
                # Add more traits as needed
            }
        }
        
        # Walk through the project directory
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith('.rs') and 'bin' not in root:  
                    file_path = os.path.join(root, file)
                    self._analyze_file(project_name, file_path)
        
        # Update cumulative stats
        # self._update_cumulative_stats(project_name)
        
        return self.project_stats[project_name]
    
    def _analyze_file(self, project_name, file_path):
        """Analyze a single Rust file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Increment file count
            self.project_stats[project_name]["total_files"] += 1
            
            # Parse the file
            tree = PARSER.parse(bytes(code, "utf-8"))
            root_node = tree.root_node
            
            # Check if it's an interface file (contains trait definitions)
            if self._has_trait_definition(root_node):
                self.project_stats[project_name]["interface_files"] += 1
            
            # Analyze custom types defined in the file
            self._analyze_custom_types(root_node, project_name, code)
            
            # Analyze functions
            functions = get_rust_functions(code)
            # deduplicate functions
            visited_funcs= set()
            final_list = []
            for f in functions:
                if f["function_name"] not in visited_funcs:
                    visited_funcs.add(f["function_name"])
                    final_list.append(f)
            functions = final_list
            self.project_stats[project_name]["functions"].extend(functions)
            total_args = [len(f["arguments"]) for f in functions]
            self.project_stats[project_name]["total_arguments"] += sum(total_args)
            # Analyze traits implemented
            self._analyze_traits_implemented(root_node, project_name)
            
            # Analyze function arguments and return types
            for func in functions:
                self._analyze_function(func, project_name)
                
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
    
    def _has_trait_definition(self, node):
        """Check if the file contains trait definitions."""
        for child in node.children:
            if child.type == "trait_item":
                return True
            # Recursively check children
            if self._has_trait_definition(child):
                return True
        return False
    
    def _analyze_custom_types(self, node, project_name, code):
        """Extract all custom type definitions from the file."""
        for child in node.children:
            # Check for struct definitions
            if child.type == "struct_item":
                struct_name_node = child.child_by_field_name("name")
                if struct_name_node:
                    struct_name = struct_name_node.text.decode("utf-8")
                    self.project_stats[project_name]["custom_types"]["structs"].add(struct_name)
            
            # Check for enum definitions
            elif child.type == "enum_item":
                enum_name_node = child.child_by_field_name("name")
                if enum_name_node:
                    enum_name = enum_name_node.text.decode("utf-8")
                    self.project_stats[project_name]["custom_types"]["enums"].add(enum_name)
            
            # Check for type aliases
            elif child.type == "type_item":
                type_name_node = child.child_by_field_name("name")
                if type_name_node:
                    type_name = type_name_node.text.decode("utf-8")
                    self.project_stats[project_name]["custom_types"]["type_aliases"].add(type_name)
            
            # Check for trait definitions
            elif child.type == "trait_item":
                trait_name_node = child.child_by_field_name("name")
                if trait_name_node:
                    trait_name = trait_name_node.text.decode("utf-8")
                    self.project_stats[project_name]["custom_types"]["traits"].add(trait_name)
            
            # Recursively analyze nested nodes
            self._analyze_custom_types(child, project_name, code)
    
    def _analyze_traits_implemented(self, node, project_name):
        """Analyze trait implementations in the file."""
        for child in node.children:
            if child.type == "impl_item":
                # Check if this is a trait implementation
                trait_clause = child.child_by_field_name("trait_clause")
                if trait_clause:
                    trait_node = trait_clause.child_by_field_name("trait")
                    if trait_node:
                        trait_name = trait_node.text.decode("utf-8")
                        # Check for specific traits
                        for trait in self.project_stats[project_name]["traits_implemented"].keys():
                            if trait in trait_name:
                                self.project_stats[project_name]["traits_implemented"][trait] += 1
            
            # Recursively check children
            self._analyze_traits_implemented(child, project_name)
    
    def _analyze_function(self, func, project_name):
        """Analyze a function's arguments and return type."""
        # Check arguments
        mut_flag, lifetime_flag, pointer_flag, borrow_flag = False, False, False, False
        for arg in func["arguments"]:
            arg_type = arg["type"]
            
            # Check for borrowed arguments
            if arg_type.startswith('&'):
                borrow_flag = True
            
            # Check for custom types
            if self._is_custom_type(arg_type):
                self.project_stats[project_name]["custom_arg_types"].add(arg_type)
            
            if 'mut' in arg_type:
                mut_flag = True
            if "'a" in arg_type:
                lifetime_flag = True
            
            if 'Box<' in arg_type or 'Rc<' in arg_type or 'Arc<' in arg_type:
                pointer_flag = True
        if borrow_flag:
            self.project_stats[project_name]["borrowed_arguments"] += 1
        if mut_flag:
            self.project_stats[project_name]["mut_arguments"] += 1
        if lifetime_flag:
            self.project_stats[project_name]["lifetime_annotations"] += 1
        if pointer_flag:
            self.project_stats[project_name]["pointer_like_types"] += 1
        
            
        
        # Check return type
        if func["return_type"] and self._is_custom_type(func["return_type"]):
            self.project_stats[project_name]["custom_return_types"].add(func["return_type"])
    
    def _is_custom_type(self, type_str):
        """
        Check if a type is a custom type (not a built-in Rust type).
        Handles complex cases like Vec<Vec<T>> and &[u8; size].
        """
        # Remove leading & for reference types
        is_reference = type_str.startswith('&')
        if is_reference:
            type_str = type_str[1:].lstrip()
        
        # Handle raw pointers
        type_str = type_str.replace('*const ', '').replace('*mut ', '')

        # handle reference types
        type_str = type_str.replace('&', '')

        # Handle mutable references
        type_str = type_str.replace("mut ", "")
        
        # Handle array types like [u8; 32]
        array_match = re.match(r'\[(.*?);.*]', type_str)
        if array_match:
            inner_type = array_match.group(1).strip()
            return self._is_custom_type(inner_type)
        
        # List of built-in Rust types
        built_in_types = [
            'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
            'u8', 'u16', 'u32', 'u64', 'u128', 'usize',
            'f32', 'f64', 'bool', 'char', 'str', 'String',
            'usize',
            'Vec', 'Option', 'Result', 'Box', 'Rc', 'Arc',
            'HashMap', 'HashSet', 'BTreeMap', 'BTreeSet',
            'VecDeque', 'LinkedList', 'BinaryHeap',
            'Range', 'RangeInclusive', 'RangeFull',
            'Path', 'PathBuf',
            'Cow', 'Mutex', 'RwLock', 'Once', 'thread',
            '()', 'Self', 'self', 'dyn'
        ]
        if 'u8' in type_str:
            return False
        
        # Handle generic types recursively
        if '<' in type_str and '>' in type_str:
            # Extract base type and generic parameters
            base_type = type_str[:type_str.find('<')].strip()
            
            # Extract parameters between outermost angle brackets
            bracket_depth = 0
            start_index = type_str.find('<')
            end_index = -1
            
            for i in range(start_index, len(type_str)):
                if type_str[i] == '<':
                    bracket_depth += 1
                elif type_str[i] == '>':
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        end_index = i
                        break
            
            inner_content = type_str[start_index+1:end_index]
            
            # Check if base type is built-in
            if any(base_type == bt for bt in built_in_types):
                # Split parameters by comma, but respect nested generics
                params = []
                current_param = ""
                param_bracket_depth = 0
                
                for char in inner_content + ',':  # Add comma to process last param
                    if char == ',' and param_bracket_depth == 0:
                        params.append(current_param.strip())
                        current_param = ""
                    else:
                        if char == '<':
                            param_bracket_depth += 1
                        elif char == '>':
                            param_bracket_depth -= 1
                        current_param += char
                
                # Check if any parameter is a custom type
                for param in params:
                    if self._is_custom_type(param):
                        return True
                
                return False
            
            return True  # Custom generic type
        
        # Handle tuples
        if type_str.startswith('(') and type_str.endswith(')'):
            inner_tuple = type_str[1:-1]
            
            # Split by commas, respecting nested types
            params = []
            current_param = ""
            param_bracket_depth = 0
            
            for char in inner_tuple + ',':  # Add comma to process last param
                if char == ',' and param_bracket_depth == 0:
                    params.append(current_param.strip())
                    current_param = ""
                else:
                    if char in '<(':
                        param_bracket_depth += 1
                    elif char in '>)':
                        param_bracket_depth -= 1
                    current_param += char
            
            # Check if any tuple element is a custom type
            for param in params:
                if param and self._is_custom_type(param):
                    return True
            
            return False
        
        # Check if type is a direct match or starts with a built-in type
        for bt in built_in_types:
            if type_str == bt or type_str.startswith(bt + '::'):
                return False
        
        # If we've ruled out all built-in cases, it's likely a custom type
        return True
    
    def _update_cumulative_stats(self, project_name):
        """Update cumulative stats with the current project's stats."""
        proj_stats = self.project_stats[project_name]
        
        self.cumulative_stats["total_files"] += proj_stats["total_files"]
        self.cumulative_stats["interface_files"] += proj_stats["interface_files"]
        self.cumulative_stats["total_functions"] += len(proj_stats["functions"])
        self.cumulative_stats["borrowed_arguments"] += proj_stats["borrowed_arguments"]
        self.cumulative_stats["custom_arg_types"].update(proj_stats["custom_arg_types"])
        self.cumulative_stats["custom_return_types"].update(proj_stats["custom_return_types"])
        self.cumulative_stats["pointers"] += proj_stats["pointers"]
        
        # Update custom types
        for type_category in ["structs", "enums", "type_aliases", "traits"]:
            self.cumulative_stats["custom_types"][type_category].update(
                proj_stats["custom_types"][type_category]
            )
        
        # Update trait implementations
        for trait, count in proj_stats["traits_implemented"].items():
            self.cumulative_stats["traits_implemented"][trait] += count
    
    def analyze_multiple_projects(self, project_paths):
        """Analyze multiple Rust projects and generate cumulative stats."""
        for project_path in tqdm(project_paths, desc="Analyzing projects"):
            self.analyze_project(project_path)
        
        return self.generate_report()
    
    def generate_report(self):
        
        """Generate a comprehensive report of project and cumulative stats."""
        report = {
            "project_stats": {},
            "cumulative_stats": {
                "total_projects": len(self.project_stats),
                "total_files": self.cumulative_stats["total_files"],
                "interface_files": self.cumulative_stats["interface_files"],
                "total_functions": self.cumulative_stats["total_functions"],
                "total_args": self.cumulative_stats["total_arguments"],
                "borrowed_arguments": self.cumulative_stats["borrowed_arguments"],
                "custom_arg_types": list(self.cumulative_stats["custom_arg_types"]),
                "custom_arg_types_count": len(self.cumulative_stats["custom_arg_types"]),
                "custom_return_types": list(self.cumulative_stats["custom_return_types"]),
                "custom_return_types_count": len(self.cumulative_stats["custom_return_types"]),
                "pointers": self.cumulative_stats["pointers"],
                "custom_types": {
                    "structs": list(self.cumulative_stats["custom_types"]["structs"]),
                    "structs_count": len(self.cumulative_stats["custom_types"]["structs"]),
                    "enums": list(self.cumulative_stats["custom_types"]["enums"]),
                    "enums_count": len(self.cumulative_stats["custom_types"]["enums"]),
                    "type_aliases": list(self.cumulative_stats["custom_types"]["type_aliases"]),
                    "type_aliases_count": len(self.cumulative_stats["custom_types"]["type_aliases"]),
                    "traits": list(self.cumulative_stats["custom_types"]["traits"]),
                    "traits_count": len(self.cumulative_stats["custom_types"]["traits"])
                },
                "traits_implemented": self.cumulative_stats["traits_implemented"]
            }
        }
        
        # Process individual project stats for the report
        for project_name, stats in self.project_stats.items():
            percentage_functions_with_borrowed_args = 0
            percentage_functions_custom_type = 0
            percentage_functions_with_custom_return_type = 0
            for f in self.project_stats[project_name]["functions"]:
                if any(['&' in ff["type"] for ff in f["arguments"]]):
                    percentage_functions_with_borrowed_args += 1
                if any([self._is_custom_type(ff["type"]) for ff in f["arguments"]]):
                    percentage_functions_custom_type += 1
                if self._is_custom_type(f["return_type"]):
                    percentage_functions_with_custom_return_type += 1
            if len(stats["functions"]) == 0:
                print(project_name)
            else:
                percentage_function_with_mut = stats["mut_arguments"] / len(stats["functions"]) * 100
                percentage_functions_with_lifetime = stats["lifetime_annotations"] / len(stats["functions"]) * 100
                percentage_functions_with_pointer_like_types = stats["pointer_like_types"] / len(stats["functions"]) * 100
                percentage_functions_with_borrowed_args = percentage_functions_with_borrowed_args / len(self.project_stats[project_name]["functions"]) * 100
                percentage_functions_custom_type = percentage_functions_custom_type / len(self.project_stats[project_name]["functions"]) * 100
                percentage_functions_with_custom_return_type = percentage_functions_with_custom_return_type / len(self.project_stats[project_name]["functions"]) * 100
                    
                report["project_stats"][project_name] = {
                    "total_files": stats["total_files"],
                    "interface_files": stats["interface_files"],
                    "total_functions": len(stats["functions"]),
                    "total_args": stats["total_arguments"],
                    "borrowed_arguments": stats["borrowed_arguments"],
                    "custom_arg_types": list(stats["custom_arg_types"]),
                    "custom_arg_types_count": len(stats["custom_arg_types"]),
                    "custom_return_types": list(stats["custom_return_types"]),
                    "custom_return_types_count": len(stats["custom_return_types"]),
                    "pointers": stats["pointers"],
                    "percentage_functions_with_borrowed_args": percentage_functions_with_borrowed_args,
                    "percentage_functions_custom_type": percentage_functions_custom_type,
                    "percentage_functions_with_custom_return_type": percentage_functions_with_custom_return_type,
                    "percentage_functions_with_mut": percentage_function_with_mut,
                    "percentage_functions_with_lifetime": percentage_functions_with_lifetime,
                    "percentage_functions_with_pointer_like_types": percentage_functions_with_pointer_like_types,
                    "custom_types": {
                        "structs": list(stats["custom_types"]["structs"]),
                        "structs_count": len(stats["custom_types"]["structs"]),
                        "enums": list(stats["custom_types"]["enums"]),
                        "enums_count": len(stats["custom_types"]["enums"]),
                        "type_aliases": list(stats["custom_types"]["type_aliases"]),
                        "type_aliases_count": len(stats["custom_types"]["type_aliases"]),
                        "traits": list(stats["custom_types"]["traits"]),
                        "traits_count": len(stats["custom_types"]["traits"])
                    },
                    "traits_implemented": stats["traits_implemented"]
                }
            
        return report
    
    def save_report(self, output_path):
        """Save the analysis report to a JSON file."""
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {output_path}")
    
    def generate_visualization(self, output_dir):
        """Generate visualization charts for the analysis results."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Custom Types Distribution
            custom_types = self.cumulative_stats["custom_types"]
            types_counts = [
                len(custom_types["structs"]),
                len(custom_types["enums"]),
                len(custom_types["type_aliases"]),
                len(custom_types["traits"])
            ]
            types_labels = ['Structs', 'Enums', 'Type Aliases', 'Traits']
            
            plt.figure(figsize=(10, 6))
            plt.bar(types_labels, types_counts, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
            plt.title('Distribution of Custom Types')
            plt.ylabel('Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, 'custom_types_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Traits Implementation
            traits = self.cumulative_stats["traits_implemented"]
            traits_labels = list(traits.keys())
            traits_counts = list(traits.values())
            
            plt.figure(figsize=(12, 6))
            plt.bar(traits_labels, traits_counts, color='#9b59b6')
            plt.title('Trait Implementations')
            plt.ylabel('Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, 'trait_implementations.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Project Comparison (for multiple projects)
            if len(self.project_stats) > 1:
                projects = list(self.project_stats.keys())
                functions = [len(self.project_stats[p]["functions"]) for p in projects]
                structs = [len(self.project_stats[p]["custom_types"]["structs"]) for p in projects]
                
                x = np.arange(len(projects))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(14, 8))
                rects1 = ax.bar(x - width/2, functions, width, label='Functions')
                rects2 = ax.bar(x + width/2, structs, width, label='Structs')
                
                ax.set_title('Functions vs Structs by Project')
                ax.set_xticks(x)
                ax.set_xticklabels(projects, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                fig.tight_layout()
                plt.savefig(os.path.join(output_dir, 'project_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Visualizations saved to {output_dir}")
            
        except ImportError:
            print("Matplotlib is required for visualization. Install with 'pip install matplotlib'")



def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <project_paths_file> [output_file] [--visualize <output_dir>]")
        return
    
    # File containing list of project paths, one per line
    projects_file = sys.argv[1]
    output_file = "./rust_analysis_report.json"
    visualize = False
    output_dir = "visualizations"
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    if len(sys.argv) > 3 and sys.argv[3] == "--visualize":
        visualize = True
        if len(sys.argv) > 4:
            output_dir = sys.argv[4]
    
    # Read project paths
    project_paths = []
    for p in Path(projects_file).iterdir():
        project_paths.append(p)
    
    # Initialize analyzer
    analyzer = RustProjectAnalyzer()
    
    # Analyze projects
    analyzer.analyze_multiple_projects(project_paths)
    
    # Save report
    analyzer.save_report(output_file)
    
    # Generate visualizations if requested
    if visualize:
        analyzer.generate_visualization(output_dir)

if __name__ == "__main__":
    main()