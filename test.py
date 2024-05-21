# Read output file
output_file = "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/output.txt"
with open(output_file, "r") as f:
    lines = f.readlines()

# Find the line with prices
prices_line = None
for i, line in enumerate(lines):
    if line.strip() == "Prices:":
        prices_line = lines[i+1]
        break

# Parse prices
prices = eval(prices_line)

# Print the length of the prices list
print(len(prices))