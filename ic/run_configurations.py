import subprocess
import os
from itertools import product

# Define the parameter values to vary
BETA_values = [50] #, 100, 1000, 10000]
dropout_good_valuation_values = [40]
default_good_valuation_values = [1]
price_default_good_values = [1]
lambda_frequency_values = [10, 30]
price_upper_bound_values = [10000]
# beta_adjustment_methods = ["pidcontrol", "adjustedlearning", "errorbased", "normalizederror", "none"]
beta_adjustment_methods = ["none"]
# num_agents_to_run = [2,5,10,15,20,30]
num_agents_to_run = [40]
tol_error_to_check = [0.1, 0.01, 0.001]
alpha_values = [1]
# alpha_values = [1.0, 0.5, 0.1, 5.0, 10.0]
# num_agents_to_run = [170]
# num_CPUS = 10


# Generate all combinations of the parameter values
# "--file", "test_cases/casef_20240614_153258.json",
parameter_combinations = list(product(BETA_values, dropout_good_valuation_values, default_good_valuation_values, price_default_good_values, lambda_frequency_values, 
                                      price_upper_bound_values, num_agents_to_run, beta_adjustment_methods, alpha_values))
main_script_path = os.path.join(os.path.dirname(__file__), 'main.py')


# file_list = ["test_cases/toulouse_case_cap10_updated_20stepauction_15sectau.json"]
file_list = ["test_cases/toulouse_case_cap7_updated_20.json"]
# file_list = ["test_cases/toulouse_case_cap4_updated.json"]

for file in file_list:
    for idx, (BETA, dropout_good_valuation, default_good_valuation, price_default_good, 
              lambda_frequency, price_upper_bound, num_agents_to_run, beta_adjustment_method, alpha) in enumerate(parameter_combinations):
        args = [
            "python", main_script_path,
            "--file", file,
            # "--file", "test_cases/archived_presub/small_receding_toulouse_case_withC.json",
            # "--file", "test_cases/archived_presub/casef_20250109_174256.json",
            # "--file", "test_cases/casef_20250115_205310.json",
            # "--file", "test_cases/casef_20250116_153516.json",
            # "--file", "test_cases/toulouse_case.json",
            "--method", "fisher",
            "--output_bsky", str(False),
            "--force_overwrite",
            "--BETA", str(BETA),
            "--dropout_good_valuation", str(dropout_good_valuation),
            "--default_good_valuation", str(default_good_valuation),
            "--price_default_good", str(price_default_good),
            "--lambda_frequency", str(lambda_frequency),
            "--price_upper_bound", str(price_upper_bound),
            # "--num_agents_to_run", str(num_agents_to_run),
            "--run_up_to_auction", str(1000),
            "--save_pkl_files", "True",
            "--beta_adjustment_method", beta_adjustment_method,
            "--alpha", str(alpha),
            "--use_AADMM", "False",
            "--tol_error_to_check"
        ] + [str(tol) for tol in tol_error_to_check]
            
        

        print(f"Running configuration {idx + 1}/{len(parameter_combinations)}: "
            f"BETA={BETA}, dropout_good_valuation={dropout_good_valuation}, "
            f"default_good_valuation={default_good_valuation}, price_default_good={price_default_good}, "
            f"lambda_frequency={lambda_frequency}, price_upper_bound={price_upper_bound}, beta_adjustment_method={beta_adjustment_method}")
        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Configuration {idx + 1} failed: {e}. Skipping to the next configuration.")