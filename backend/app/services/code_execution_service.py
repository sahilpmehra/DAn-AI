import subprocess
import tempfile
import os
import pandas as pd
from typing import Dict, Any
import json
import timeout_decorator
import ast
import venv
import sys
from pathlib import Path

class CodeExecutionService:
    TIMEOUT_SECONDS = 30
    ALLOWED_MODULES = {'pandas', 'numpy', 'sklearn', 'json', 'StringIO'}
    
    def __init__(self):
        self.sandbox_path = None
        self.python_executable = None
        self.setup_sandbox_environment()
    
    def setup_sandbox_environment(self):
        """Setup a sandboxed virtual environment with only required packages"""
        try:
            # Create a temporary directory for the virtual environment
            sandbox_dir = tempfile.mkdtemp(prefix='code_sandbox_')
            self.sandbox_path = sandbox_dir
            
            # Create virtual environment
            venv.create(sandbox_dir, with_pip=True)
            
            # Set python executable path based on OS
            if sys.platform == 'win32':
                self.python_executable = os.path.join(sandbox_dir, 'Scripts', 'python.exe')
            else:
                self.python_executable = os.path.join(sandbox_dir, 'bin', 'python')
            
            # Install required packages
            subprocess.run([
                self.python_executable, 
                '-m', 'pip', 
                'install', 
                'pandas==2.2.1',
                'numpy==1.26.4'
            ], check=True, capture_output=True)
            
        except Exception as e:
            if self.sandbox_path and os.path.exists(self.sandbox_path):
                self._cleanup_sandbox()
            raise Exception(f"Failed to setup sandbox environment: {str(e)}")
    
    def _cleanup_sandbox(self):
        """Clean up the sandbox environment"""
        try:
            if self.sandbox_path and os.path.exists(self.sandbox_path):
                import shutil
                shutil.rmtree(self.sandbox_path)
        except Exception as e:
            print(f"Warning: Failed to cleanup sandbox: {str(e)}")
    
    def validate_code(self, code: str) -> bool:
        """Validate code for security concerns"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name.split('.')[0] not in self.ALLOWED_MODULES:
                            raise ValueError(f"Import of {name.name} not allowed")
                elif isinstance(node, ast.ImportFrom):
                    if node.module.split('.')[0] not in self.ALLOWED_MODULES:
                        raise ValueError(f"Import from {node.module} not allowed")
            return True
        except Exception as e:
            raise ValueError(f"Code validation failed: {str(e)}")

    def _clean_code_block(self, code: str) -> str:
        """Remove code block markers and clean the code"""
        # Remove ```python at the start if present
        if code.startswith('```python'):
            code = code[len('```python'):]
        # Remove ``` at the end if present
        if code.endswith('```'):
            code = code[:-3]
        # Strip any leading/trailing whitespace
        return code.strip()

    @timeout_decorator.timeout(TIMEOUT_SECONDS)
    def execute_code(self, code: str, data: pd.DataFrame, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute code in a sandboxed environment"""
        try:
            # Clean the code first
            code = self._clean_code_block(code)
            
            # Validate the code
            self.validate_code(code)
            
            # Ensure input_data is a dictionary
            input_data = input_data if input_data is not None else {}
            
            # Create temporary files for data, input_data and code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file, \
                 tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as data_file, \
                 tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
                
                # Save DataFrame temporarily
                data.to_csv(data_file.name, index=False)
                data_path = data_file.name
                
                # Save input_data temporarily
                json.dump(input_data, input_file)
                input_file.flush()  # Ensure data is written to disk
                input_path = input_file.name
                
                # Prepare the code with data loading
                full_code = f"""
import pandas as pd
import numpy as np
import json
import sys
from io import StringIO

# Load the data
df = pd.read_csv(r'{data_path}')

# Load the input data
try:
    with open(r'{input_path}', 'r') as f:
        input_data = json.load(f)
except Exception:
    input_data = {{}}

{code}

# Execute the code and capture the result
result = process_data(df, input_data)

# Output the result as JSON
json_result = json.dumps(result)
sys.stdout.write(json_result)
sys.stdout.write("\\n")
sys.stdout.flush()
"""
                code_file.write(full_code)
                code_file.flush()
                code_path = code_file.name

                try:
                    # Convert paths to absolute paths
                    abs_code_path = os.path.abspath(code_path)
                    abs_python_path = os.path.abspath(self.python_executable)
                    
                    # Execute in sandbox using the virtual environment's Python
                    result = subprocess.run(
                        [abs_python_path, abs_code_path],
                        capture_output=True,
                        text=True,
                        timeout=self.TIMEOUT_SECONDS,
                        env={"PYTHONUNBUFFERED": "1"}
                    )
                    
                    if result.returncode != 0:
                        raise Exception(f"Execution failed: {result.stderr}")
                    
                    # Strip any whitespace and ensure we have valid output
                    output = result.stdout.strip()
                    if not output:
                        if result.stderr:
                            raise Exception(f"No output received. Debug info: {result.stderr}")
                        raise Exception("No output received from code execution")
                    
                    try:
                        # Parse the JSON output into a dictionary
                        return json.loads(output)
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid JSON output: {str(e)}. Output was: {output}")
                    
                finally:
                    # Cleanup temporary files
                    os.unlink(code_path)
                    os.unlink(data_path)
                    os.unlink(input_path)
                    
        except Exception as e:
            raise Exception(f"Code execution failed: {str(e)}")
            
    def __del__(self):
        """Cleanup sandbox when the service is destroyed"""
        self._cleanup_sandbox() 