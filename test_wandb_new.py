import wandb
import os

# Ensure we're using the conda environment packages
os.environ["PYTHONNOUSERSITE"] = "1"

print("Testing wandb connection...")
try:
    # Use a simpler project name to avoid any permission issues
    project_name = "htcdc-test"
    
    # Initialize wandb with your correct username and the test project
    wandb.init(
        project=project_name,
        entity="ahmad-naghavi-ozu",
        name="connection_test"
    )
    
    print(f"Successfully connected to wandb!")
    print(f"Project: {wandb.run.project}")
    print(f"Entity: {wandb.run.entity}")
    
    # Log a simple metric
    wandb.log({"test_metric": 1.0})
    
    print("Test complete and data logged successfully.")
    print(f"Please check your wandb dashboard at: https://wandb.ai/ahmad-naghavi-ozu/{project_name}")
    
    # Finish the run
    wandb.finish()
except Exception as e:
    print(f"Error connecting to wandb: {e}")
    
print("Script execution finished.")