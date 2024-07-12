import re

def find_best_accuracy(file_path):
    best_accuracy = None
    best_line_numbers = []
    best_line_values = ""

    # Regular expression to match lines like "Best accuracy: <number>"
    pattern = r"Best accuracy:\s*([0-9]*\.?[0-9]+)"

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            accuracy = float(match.group(1))
            if best_accuracy is None or accuracy > best_accuracy:
                best_accuracy = accuracy
                best_line_numbers = [i + 1]  # Store line numbers (1-based index)
                best_line_values = lines[i-3]
            

    return best_accuracy, best_line_numbers, best_line_values

# Example usage:
file_path = 'real_results.txt'
best_accuracy, best_line_numbers, best_line_values = find_best_accuracy(file_path)

print(f"Highest Best Accuracy: {best_accuracy}")
print(f"Lines with Highest Best Accuracy: {best_line_numbers}")
print(best_line_values)
