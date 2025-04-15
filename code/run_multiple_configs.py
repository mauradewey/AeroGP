# loop through all config files and run AeroGP_SVGP_train_model.py with each one
# the config file will dictate if a new version of the model is trained, or if a previous one is loaded and tested with new output 
# (so the order of the config files is important)
# output is saved according to the config file name, and any errors are logged to a file in the error_logs directory
#
# March 2025
#
# usage: python run_multiple_configs.py

import subprocess
import os
import glob
import yaml
from pathlib import Path

config_dir = "config_files/" # Directory containing config files to run
config_files = glob.glob(config_dir + '*lr.yml')

# Path to the target script
target_script = "AeroGP_SVGP_train_model.py"

for config in config_files:
    print(f"Running {target_script} with {config}")

    try:
        # Read config file to get log_dir for saving output
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)
            
        log_dir = config_data["log_dir"]
        config = Path(config)
        output_log_file = os.path.join(log_dir, f"{config.stem}_output.log")
        error_log_file = os.path.join(log_dir, f"{config.stem}_error.log")

        # Run
        result = subprocess.run(
            ["python", "-u", target_script, config],
            capture_output=True,
            text=True,
            errors="replace",
        )
        # Save stdout (normally printed to console) to a log file
        with open(output_log_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        # If the process failed, log the error
        if result.returncode != 0:
            print(f"Error with {config}, logging to {error_log_file}")
            with open(error_log_file, "w") as f:
                f.write(f"Error running {target_script} with {config}\n")
                f.write(result.stderr)

    except Exception as e:
        print(f"Unexpected error with {config}: {e}")

    print("Done.")

print("All training/testing complete!")
