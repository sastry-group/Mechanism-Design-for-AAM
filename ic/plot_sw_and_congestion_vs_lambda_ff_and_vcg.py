import json
from matplotlib import pyplot as plt
from pathlib import Path
import sys

top_level_path = Path(__file__).resolve().parent.parent
sys.path.append(str(top_level_path))
from ic.main import run_scenario

LAMBDAS = [0.1, 0.2, 0.5, 1, 2, 5, 10]
case_file_path = "test_cases/casef_20240910_232939.json"

def load_json(file=None):
    """
    Load a case file for a bluesky simulation from a JSON file.
    """
    if file is None:
        return None
    assert Path(file).is_file(), f"File {file} does not exist."

    # Load the JSON file
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Opened file {file}")
    return data

# Read in case file
test_case_data = load_json(case_file_path)

# Iterate across lambda values
methods = ["ff", "vcg"]
results = {method: [] for method in methods}
for lambda_val in LAMBDAS:
    for method in methods:
        print(f"Running method {method} with lambda {lambda_val}")
        # Read in case file
        test_case_data["congestion_params"]["lambda"] = lambda_val

        # Run the scenario
        _, scenario_result = run_scenario(test_case_data, "", "", method, save_scenario=False, payment_calc=False)
        results[method].append(scenario_result[0])
            
# Plot
social_welfare = [[result[2] for result in method_results] for method_results in results.values()]
congestion_costs = [[result[3] for result in method_results] for method_results in results.values()]

plt.subplot(2, 1, 1)
plt.plot(LAMBDAS, social_welfare[0], "*-", label=methods[0])
plt.plot(LAMBDAS, social_welfare[1], "*-", label=methods[1])
plt.xlabel("Lambda")
plt.ylabel("Social Welfare")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(LAMBDAS, congestion_costs[0], "*-", label=methods[0])
plt.plot(LAMBDAS, congestion_costs[1], "*-", label=methods[1])
plt.xlabel("Lambda")
plt.ylabel("Congestion Costs")
plt.legend()
plt.show()
