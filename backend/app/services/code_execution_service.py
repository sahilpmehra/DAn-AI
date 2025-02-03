import subprocess
import tempfile
import os
import pandas as pd
from typing import Dict, Any
import json
import timeout_decorator
import ast

class CodeExecutionService:
    TIMEOUT_SECONDS = 30
    ALLOWED_MODULES = {'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn'}
    
    def __init__(self):
        self.setup_sandbox_environment()
    
    def setup_sandbox_environment(self):
        # Setup sandbox environment configuration
        # This is a placeholder - implement actual sandbox setup
        pass
        
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

    @timeout_decorator.timeout(TIMEOUT_SECONDS)
    def execute_code(self, code: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute code in a sandboxed environment"""
        try:
            # Create temporary files for data and code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
                # Prepare the code with data loading
                full_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import json

# Load the data
df = pd.read_csv('{data_path}')

# Execute the analysis
{code}

# Convert results to JSON
result = {{
    'data': result_data if 'result_data' in locals() else None,
    'plot': plot_data if 'plot_data' in locals() else None,
    'metadata': metadata if 'metadata' in locals() else None
}}
print(json.dumps(result))
"""
                code_file.write(full_code)
                code_path = code_file.name

            # Save DataFrame temporarily
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as data_file:
                data.to_csv(data_file.name, index=False)
                data_path = data_file.name

            # Execute in sandbox
            try:
                result = subprocess.run(
                    ['python', code_path],
                    capture_output=True,
                    text=True,
                    timeout=self.TIMEOUT_SECONDS
                )
                
                if result.returncode != 0:
                    raise Exception(f"Execution failed: {result.stderr}")
                
                return json.loads(result.stdout)
                
            finally:
                # Cleanup
                os.unlink(code_path)
                os.unlink(data_path)
                
        except Exception as e:
            raise Exception(f"Code execution failed: {str(e)}") 