import json
from matplotlib import pyplot as plt
from pathlib import Path
import sys
import pickle

top_level_path = Path(__file__).resolve().parent.parent
sys.path.append(str(top_level_path))
from ic.main import run_scenario

LAMBDAS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 100]
case_file_path = "test_cases/casef_20240911_142824.json"

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

methods = ["ff", "vcg"]
if not Path("ic/results/sw_and_congestion_vs_lambda_ff_and_vcg.pkl").is_file():
    # Read in case file
    test_case_data = load_json(case_file_path)

    # Iterate across lambda values
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

    pickle.dump((LAMBDAS, social_welfare, congestion_costs), open("ic/results/sw_and_congestion_vs_lambda_ff_and_vcg.pkl", "wb"))
else:
    LAMBDAS, social_welfare, congestion_costs = pickle.load(open("ic/results/sw_and_congestion_vs_lambda_ff_and_vcg.pkl", "rb"))

assert all([social_welfare[0][i] - congestion_costs[0][i] <= social_welfare[1][i] - congestion_costs[1][i] for i in range(len(LAMBDAS))]), "FF is not better than VCG"

plt.subplot(2, 1, 1)
plt.plot(LAMBDAS, social_welfare[0], "*-", label=methods[0])
plt.plot(LAMBDAS, social_welfare[1], "*-", label=methods[1] + "[ours]")
plt.ylim(0, 1.1* max(max(social_welfare)))
plt.xlabel("Lambda")
plt.ylabel("Social Welfare")
plt.xscale("log")
plt.legend()
plt.title("Social Welfare vs. Lambda")
plt.subplot(2, 1, 2)
plt.plot(LAMBDAS, congestion_costs[0], "*-", label=methods[0])
plt.plot(LAMBDAS, congestion_costs[1], "*-", label=methods[1] + "[ours]")
plt.ylim(0, 1.1 * max(max(congestion_costs)))
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Congestion Costs")
plt.title("Congestion Costs vs. Lambda")
plt.legend()
plt.tight_layout()
plt.show()
