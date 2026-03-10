"""
Script to automatically run Jupyter notebooks sequentially using Papermill.
This script executes all notebooks in the 'notebooks' folder in order.
"""

import os
import sys
from pathlib import Path
import papermill as pm
from datetime import datetime


def run_notebooks_with_papermill(
    notebooks_dir: str = "notebooks",
    output_dir: str = "outputs/executed_notebooks",
    kernel_name: str = None,
    parameters: dict = None,
    notebook_order: list = None,
):
    """
    Run all Jupyter notebooks in a directory sequentially using Papermill.
    
    Args:
        notebooks_dir: Directory containing notebooks to run
        output_dir: Directory to save executed notebooks
        kernel_name: Kernel to use for execution (default: None - uses notebook's default)
        parameters: Dictionary of parameters to pass to notebooks
        notebook_order: List of notebook filenames in the order to run them. 
                       If None, notebooks are sorted alphabetically.
    
    Returns:
        List of executed notebook paths with their execution status
    """
    
    # Save the original working directory
    original_cwd = os.getcwd()
    
    # Get the absolute paths
    notebooks_path = Path(notebooks_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all notebook files
    if notebook_order:
        # Use specified order
        notebook_files = [notebooks_path / name for name in notebook_order]
    else:
        # Sort alphabetically
        notebook_files = sorted(notebooks_path.glob("*.ipynb"))
    
    if not notebook_files:
        print(f"No notebooks found in {notebooks_dir}")
        return []
    
    print(f"Found {len(notebook_files)} notebook(s) to execute")
    print(f"Output will be saved to: {output_path.absolute()}")
    print("-" * 80)
    
    results = []
    start_time = datetime.now()
    
    for idx, notebook_file in enumerate(notebook_files, 1):
        notebook_name = notebook_file.name
        output_notebook = output_path / notebook_name
        
        print(f"\n[{idx}/{len(notebook_files)}] Running: {notebook_name}")
        print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Change to notebooks directory to ensure relative paths work
            os.chdir(notebooks_path)
            
            # Execute notebook using papermill
            pm.execute_notebook(
                input_path=notebook_name,  # Use relative path
                output_path=str(output_notebook),
                kernel_name=kernel_name,
                parameters=parameters,
                progress_bar=True,
                log_output=True,
            )
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            print(f"  ✓ Successfully executed: {notebook_name}")
            results.append({
                "notebook": notebook_name,
                "status": "success",
                "output_path": str(output_notebook)
            })
            
        except Exception as e:
            # Change back to original directory in case of error
            os.chdir(original_cwd)
            
            print(f"  ✗ Error executing {notebook_name}:")
            print(f"    {str(e)}")
            results.append({
                "notebook": notebook_name,
                "status": "failed",
                "error": str(e)
            })
    
    # Summary
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    # Ensure we're back in the original directory
    os.chdir(original_cwd)
    
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total notebooks: {len(notebook_files)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Total execution time: {execution_time}")
    print(f"Executed notebooks saved to: {output_path.absolute()}")
    print("=" * 80)
    
    return results


def run_specific_notebooks(
    notebook_names: list,
    output_dir: str = "outputs/executed_notebooks",
    kernel_name: str = None,
    parameters: dict = None,
):
    """
    Run specific notebooks by name.
    
    Args:
        notebook_names: List of notebook filenames to run
        output_dir: Directory to save executed notebooks
        kernel_name: Kernel to use for execution
        parameters: Dictionary of parameters to pass to notebooks
    
    Returns:
        List of executed notebook paths with their execution status
    """
    
    # Save the original working directory
    original_cwd = os.getcwd()
    
    notebooks_dir = "notebooks"
    notebooks_path = Path(notebooks_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Running {len(notebook_names)} specific notebook(s)")
    print(f"Output will be saved to: {output_path.absolute()}")
    print("-" * 80)
    
    results = []
    start_time = datetime.now()
    
    for idx, notebook_name in enumerate(notebook_names, 1):
        notebook_file = notebooks_path / notebook_name
        
        if not notebook_file.exists():
            print(f"\n[{idx}/{len(notebook_names)}] ✗ Not found: {notebook_name}")
            results.append({
                "notebook": notebook_name,
                "status": "not_found",
                "error": f"Notebook not found: {notebook_file}"
            })
            continue
        
        output_notebook = output_path / notebook_name
        
        print(f"\n[{idx}/{len(notebook_names)}] Running: {notebook_name}")
        print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Change to notebooks directory to ensure relative paths work
            os.chdir(notebooks_path)
            
            # Execute notebook using papermill
            pm.execute_notebook(
                input_path=notebook_name,  # Use relative path
                output_path=str(output_notebook),
                kernel_name=kernel_name,
                parameters=parameters,
                progress_bar=True,
                log_output=True,
            )
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            print(f"  ✓ Successfully executed: {notebook_name}")
            results.append({
                "notebook": notebook_name,
                "status": "success",
                "output_path": str(output_notebook)
            })
            
        except Exception as e:
            # Change back to original directory in case of error
            os.chdir(original_cwd)
            
            print(f"  ✗ Error executing {notebook_name}:")
            print(f"    {str(e)}")
            results.append({
                "notebook": notebook_name,
                "status": "failed",
                "error": str(e)
            })
    
    # Summary
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    # Ensure we're back in the original directory
    os.chdir(original_cwd)
    
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total notebooks: {len(notebook_names)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Not found: {sum(1 for r in results if r['status'] == 'not_found')}")
    print(f"Total execution time: {execution_time}")
    print(f"Executed notebooks saved to: {output_path.absolute()}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Example 1: Run all notebooks in the 'notebooks' folder in specified order
    print("Starting Papermill Notebook Executor")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Specify the order of notebooks to run
    notebook_order = [
        "01_data_source.ipynb",
        "1.5_preprocessing.ipynb",
        "02_preprocess_feature.ipynb",
        "03_apriori_modelling.ipynb",
        "04_cluster_attrition_rules.ipynb",
        "05.1_modeling_xgb_train.ipynb",
        "05.2_modeling_rf_train.ipynb",
        "06_Evaluation_and_Results (1).ipynb",
        "07_Semi_SuperviseLearning.ipynb",
    ]
    
    results = run_notebooks_with_papermill(notebook_order=notebook_order)
    
    # Example 2: Run all notebooks in alphabetical order (uncomment to use)
    # results = run_notebooks_with_papermill()
    
    # Example 3: Run specific notebooks (uncomment to use)
    # results = run_specific_notebooks([
    #     "01_data_source.ipynb",
    #     "02_preprocess_feature.ipynb",
    #     "03_apriori_modelling.ipynb",
    # ])
    
    # Exit with appropriate code based on results
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    sys.exit(1 if failed_count > 0 else 0)
