import subprocess
import os
from itertools import product

# Define the parameter values to vary
BETA_values = [100,50,10,5,1]
dropout_good_valuation_values = [1]
default_good_valuation_values = [1]
price_default_good_values = [10]
lambda_frequency_values = [2,10]
price_upper_bound_values = [100]

# Generate all combinations of the parameter values
# "--file", "test_cases/casef_20240614_153258.json",
parameter_combinations = list(product(BETA_values, dropout_good_valuation_values, default_good_valuation_values, price_default_good_values, lambda_frequency_values, price_upper_bound_values))

main_script_path = os.path.join(os.path.dirname(__file__), 'main.py')

# Iterate through each combination and run the main script
for idx, (BETA, dropout_good_valuation, default_good_valuation, price_default_good, lambda_frequency, price_upper_bound) in enumerate(parameter_combinations):
    args = [
        "python", main_script_path,
        "--file", "test_cases/casef_20240925_175552_short.json",
        "--method", "fisher",
        "--force_overwrite",
        "--BETA", str(BETA),
        "--dropout_good_valuation", str(dropout_good_valuation),
        "--default_good_valuation", str(default_good_valuation),
        "--price_default_good", str(price_default_good),
        "--lambda_frequency", str(lambda_frequency),
        "--price_upper_bound", str(price_upper_bound)
    ]

    print(f"Running configuration {idx + 1}/{len(parameter_combinations)}: "
          f"BETA={BETA}, dropout_good_valuation={dropout_good_valuation}, "
          f"default_good_valuation={default_good_valuation}, price_default_good={price_default_good}, "
          f"lambda_frequency={lambda_frequency}, price_upper_bound={price_upper_bound}")
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Configuration {idx + 1} failed: {e}. Skipping to the next configuration.")