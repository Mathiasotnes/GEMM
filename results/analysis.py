#!/usr/bin/env python3
#*************************************************************************************** #
# analysis.py                                                                            #
# -------------------------------------------------------------------------------------- #
# This script reads the CSV file "gemm_benchmark_results.csv", and prints summary stats. #
# It shows a table of sizes vs methods, highlighting the fastest time in green.          #
# -------------------------------------------------------------------------------------- #
# Author: Mathias Otnes                                                                  #
# Year:   2024                                                                           #
#*************************************************************************************** #

import csv
from collections import defaultdict

# *************************************************************************************** #
# Configuration                                                                           #
# *************************************************************************************** #

FILE_NAME = "results_old.csv"

GREEN = "\033[32m"
RESET = "\033[0m"

# *************************************************************************************** #
# Implementation                                                                          #
# *************************************************************************************** #

def load_data(filename):
    data = defaultdict(list)
    methods_set = set()
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = int(row['Size'])
            method = row['Method']
            time = float(row['Time(ms)'])
            data[size].append((method, time))
            methods_set.add(method)
    return data, sorted(methods_set)

def print_table(data, methods):
    method_col_width = max(len(m) for m in methods)
    size_col_width = 4
    time_col_width = 10

    header = f"| {'Size'.ljust(size_col_width)} | " + "  | ".join(m.ljust(method_col_width) for m in methods) + "  |"
    line_length = len(header)
    divider = "-" * line_length

    print(divider)
    print(header)
    print(divider)

    # Print rows
    for size in sorted(data.keys()):
        entries = data[size]
        fastest_time = min(entries, key=lambda x: x[1])[1]
        row = f"| {str(size).ljust(size_col_width)} | "
        times_str = []
        method_to_time = dict(entries)
        for m in methods:
            t = method_to_time[m]
            time_str = f"{t:.3f}".rjust(time_col_width)
            if abs(t - fastest_time) < 1e-9:
                time_str = f"{GREEN}{time_str}{RESET}"
            times_str.append(time_str)
        row += " | ".join(times_str) + " |"
        print(row)

    print(divider)

# *************************************************************************************** #
# Main program entry point                                                                #
# *************************************************************************************** #

if __name__ == "__main__":
    data, methods = load_data(FILE_NAME)
    print_table(data, methods)
    exit(0)
