import json
from matplotlib import pyplot as plt
from pathlib import Path
import sys
import pickle

top_level_path = Path(__file__).resolve().parent.parent
sys.path.append(str(top_level_path))
from ic.main import run_scenario

LAMBDAS = [1, 2, 5, 10, 100]
case_file_path = "test_cases/case2.json"

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
    valuations = [[result[2] for result in method_results] for method_results in results.values()]
    congestion_costs = [[result[3] for result in method_results] for method_results in results.values()]
    plotted_congestion = [[-result[3]/lambda_val for result, lambda_val in zip(method_results, LAMBDAS)] for method_results in results.values()]

    pickle.dump((LAMBDAS, valuations, plotted_congestion, congestion_costs), open("ic/results/sw_and_congestion_vs_lambda_ff_and_vcg.pkl", "wb"))
else:
    LAMBDAS, valuations, plotted_congestion, congestion_costs = pickle.load(open("ic/results/sw_and_congestion_vs_lambda_ff_and_vcg.pkl", "rb"))

print(f"FF Valuations: {valuations[0]}, Congestion costs: {congestion_costs[0]}")
print(f"VCG Valuations: {valuations[1]}, Congestion costs: {congestion_costs[1]}")
print(f"FF SW: {[valuations[0][i] - congestion_costs[0][i] for i in range(len(LAMBDAS))]}")
print(f"VCG SW: {[valuations[1][i] - congestion_costs[1][i] for i in range(len(LAMBDAS))]}")
assert all([valuations[0][i] - congestion_costs[0][i] <= valuations[1][i] - congestion_costs[1][i] for i in range(len(LAMBDAS))]), "FF is not better than VCG"


sizes = [size**3 for size in list(range(1, len(LAMBDAS)+1))]
plt.scatter(plotted_congestion[0], valuations[0], sizes=sizes, color="blue")
plt.scatter(plotted_congestion[1], valuations[1], sizes=sizes, color="orange")
plt.plot(plotted_congestion[0], valuations[0], "--", color="blue", label="FF")
plt.plot(plotted_congestion[1], valuations[1], "--", color="orange", label="VCG [ours]")
# plt.ylim(0, 1.1* max(max(valuations)))
# plt.xlim(1.1* max(max(congestion_costs)), 0)
# plt.xscale("log")
for i, txt in enumerate(LAMBDAS):
    plt.annotate(r'$\lambda$=' + str(txt), (plotted_congestion[1][i] + 10, valuations[1][i] + 10))
plt.annotate("all", (plotted_congestion[0][i] + 10 , valuations[0][i] + 10))
plt.ylabel("Sum of Valuations")
plt.xlabel("Congestion Costs")
plt.title("Sum of Valuations vs. Congestion Costs")
plt.legend()
plt.show()
