import subprocess
import os
from itertools import product

# Define the parameter values to vary
BETA_values = [100, 20, 50, 60, 10]
dropout_good_valuation_values = [1]
default_good_valuation_values = [1]
price_default_good_values = [10]
rebate_frequency_values = [2]

# Generate all combinations of the parameter values
# "--file", "test_cases/casef_20240614_153258.json",
parameter_combinations = list(product(BETA_values, dropout_good_valuation_values, default_good_valuation_values, price_default_good_values, rebate_frequency_values))

main_script_path = os.path.join(os.path.dirname(__file__), 'main.py')

# Iterate through each combination and run the main script
for idx, (BETA, dropout_good_valuation, default_good_valuation, price_default_good, rebate_frequency) in enumerate(parameter_combinations):
    args = [
        "python", main_script_path,
        "--file", "test_cases/casef_20240917_081204.json",
        "--method", "fisher",
        "--force_overwrite",
        "--BETA", str(BETA),
        "--dropout_good_valuation", str(dropout_good_valuation),
        "--default_good_valuation", str(default_good_valuation),
        "--price_default_good", str(price_default_good),
        "--rebate_frequency", str(rebate_frequency)
    ]
    print(f"Running configuration {idx + 1}/{len(parameter_combinations)}: BETA={BETA}, dropout_good_valuation={dropout_good_valuation}, default_good_valuation={default_good_valuation}, price_default_good={price_default_good}")
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Configuration {idx + 1} failed: {e}. Skipping to the next configuration.")