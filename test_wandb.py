import wandb
import os

# Ensure we're using the conda environment packages
os.environ["PYTHONNOUSERSITE"] = "1"

print("Testing wandb connection...")
try:
    # Initialize wandb with your username and project
    wandb.init(
        project="HTCDC_DFC2023S_Mini_FullRes",
        entity="ahmad-naghavi-ozu",
        name="connection_test"
    )
    
    # Log a simple metric
    wandb.log({"test_metric": 1.0})
    
    print("Successfully connected to wandb!")
    print(f"Project: {wandb.run.project}")
    print(f"Entity: {wandb.run.entity}")
    
    # Finish the run
    wandb.finish()
except Exception as e:
    print(f"Error connecting to wandb: {e}")
    
print("Test complete.")