import os
import platform
import time
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog

try:
    import highspy
except ImportError:
    highspy = None

try:
    import gurobipy as gp
except ImportError:
    gp = None

try:
    import cvxpy as cp
except ImportError:
    cp = None


@dataclass
class Knapsack:
    profits: np.ndarray
    bounds: np.ndarray
    budget: float
    method: str = "sort"

    def solve(self):
        profits = np.asarray(self.profits, dtype=float)
        bounds = np.asarray(self.bounds, dtype=float)
        budget = float(self.budget)

        if self.method == "sort":
            return self._solve_sort(profits, bounds, budget)
        if self.method == "select":
            return self._solve_select(profits, bounds, budget)

        raise ValueError(f"Unknown knapsack method: {self.method}")

    @staticmethod
    def _solve_sort(profits, bounds, budget):
        order = np.argsort(-profits, kind="stable")
        x = np.zeros_like(bounds, dtype=float)
        remaining = budget

        for i in order:
            if remaining <= 0.0:
                break
            amount = min(bounds[i], remaining)
            x[i] = amount
            remaining -= amount

        return x

    @staticmethod
    def _solve_select(profits, bounds, budget):
        x = np.zeros_like(bounds, dtype=float)
        idx = np.arange(profits.size)
        remaining = budget

        while idx.size and remaining > 0.0:
            p = profits[idx]
            b = bounds[idx]

            pivot = np.partition(p, idx.size // 2)[idx.size // 2]
            high = p > pivot
            equal = p == pivot

            mass_high = b[high].sum()
            mass_equal = b[equal].sum()

            if remaining < mass_high:
                idx = idx[high]
                continue

            if remaining <= mass_high + mass_equal:
                x[idx[high]] = b[high]
                left = remaining - mass_high
                b_equal = b[equal]
                before = np.cumsum(b_equal) - b_equal
                x[idx[equal]] = np.clip(left - before, 0.0, b_equal)
                return x

            take = high | equal
            x[idx[take]] = b[take]
            remaining -= mass_high + mass_equal
            idx = idx[~take]

        return x


def make_knapsack_dual(method):
    def solve(distance_supply, distance_demand, supply, demand, lbd, gam):
        beta_max = float(min(supply.sum(), demand.sum()))

        u = Knapsack(
            profits=lbd - distance_supply,
            bounds=supply,
            budget=beta_max,
            method=method,
        ).solve()

        v = Knapsack(
            profits=gam - distance_demand,
            bounds=demand,
            budget=beta_max,
            method=method,
        ).solve()

        return u, v

    return solve


def reduced_lp_data(distance_supply, distance_demand, supply, demand, lbd, gam):
    n = supply.size
    c = np.concatenate((distance_supply - lbd, distance_demand - gam))
    upper = np.concatenate((supply, demand)).astype(float)
    balance = np.concatenate((np.ones(n), -np.ones(n)))
    return n, c, upper, balance


def split_uv(x, n):
    x = np.asarray(x, dtype=float)
    return x[:n], x[n:]


def solve_linprog(data, method):
    n, c, upper, balance = reduced_lp_data(*data)
    bounds = [(0.0, float(v)) for v in upper]

    result = linprog(
        c=c,
        A_eq=balance.reshape(1, -1),
        b_eq=np.array([0.0]),
        bounds=bounds,
        method=method,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return split_uv(result.x, n)


def solve_linprog_auto(*data):
    return solve_linprog(data, "highs")


def solve_linprog_ds(*data):
    return solve_linprog(data, "highs-ds")


def solve_linprog_ipm(*data):
    return solve_linprog(data, "highs-ipm")


def solve_highspy(data, solver_name):
    n, c, upper, balance = reduced_lp_data(*data)
    n_var = 2 * n

    lp = highspy.HighsLp()
    lp.num_col_ = n_var
    lp.num_row_ = 1
    lp.col_cost_ = c
    lp.col_lower_ = np.zeros(n_var)
    lp.col_upper_ = upper
    lp.row_lower_ = np.array([0.0])
    lp.row_upper_ = np.array([0.0])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.arange(n_var + 1, dtype=np.int32)
    lp.a_matrix_.index_ = np.zeros(n_var, dtype=np.int32)
    lp.a_matrix_.value_ = balance

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    model.setOptionValue("solver", solver_name)
    model.passModel(lp)
    model.run()

    if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(str(model.getModelStatus()))

    x = np.asarray(model.getSolution().col_value, dtype=float)
    return split_uv(x, n)


def solve_highspy_ds(*data):
    return solve_highspy(data, "simplex")


def solve_highspy_ipm(*data):
    return solve_highspy(data, "ipm")


def solve_gurobi(*data):
    n, c, upper, balance = reduced_lp_data(*data)
    n_var = 2 * n

    model = gp.Model()
    model.Params.OutputFlag = 0
    x = model.addMVar(n_var, lb=0.0, ub=upper, obj=c)
    model.addConstr(balance @ x == 0.0)
    model.ModelSense = gp.GRB.MINIMIZE
    model.optimize()

    if model.Status != gp.GRB.OPTIMAL:
        status = model.Status
        model.dispose()
        raise RuntimeError(f"Gurobi status {status}")

    solution = np.asarray(x.X, dtype=float)
    model.dispose()
    return split_uv(solution, n)


def solve_cvxpy(data, solver_name):
    n, c, upper, balance = reduced_lp_data(*data)
    x = cp.Variable(2 * n)

    problem = cp.Problem(
        cp.Minimize(c @ x),
        [balance @ x == 0.0, x >= 0.0, x <= upper],
    )
    problem.solve(solver=solver_name, verbose=False)

    if x.value is None:
        raise RuntimeError(f"{solver_name}: no solution")

    return split_uv(x.value, n)


def solve_cvxpy_clarabel(*data):
    return solve_cvxpy(data, cp.CLARABEL)


def solve_cvxpy_scs(*data):
    return solve_cvxpy(data, cp.SCS)


def build_solver_registry():
    solvers = {
        "KD-Sort": make_knapsack_dual("sort"),
        "KD-Select": make_knapsack_dual("select"),
        "linprog-Auto": solve_linprog_auto,
        "linprog-DualSimplex": solve_linprog_ds,
        "linprog-IPM": solve_linprog_ipm,
    }

    if highspy is not None:
        solvers["highspy-DualSimplex"] = solve_highspy_ds
        solvers["highspy-IPM"] = solve_highspy_ipm

    if gp is not None:
        solvers["Gurobi"] = solve_gurobi

    if cp is not None:
        installed = set(cp.installed_solvers())
        if "CLARABEL" in installed:
            solvers["CVXPY-CLARABEL"] = solve_cvxpy_clarabel
        if "SCS" in installed:
            solvers["CVXPY-SCS"] = solve_cvxpy_scs

    return solvers


SOLVERS = build_solver_registry()
REFERENCE = "KD-Sort"


def objective_value(u, v, distance_supply, distance_demand, supply, demand, lbd, gam):
    return float(
        np.dot(distance_supply, u)
        + np.dot(distance_demand, v)
        + np.dot(lbd, supply - u)
        + np.dot(gam, demand - v)
    )


def generate_dataset(n, seed):
    rng = np.random.default_rng(seed)

    points_supply = rng.integers(-100, 101, size=(n, 2))
    points_demand = rng.integers(-100, 101, size=(n, 2))
    supply = rng.integers(50, 120, size=n).astype(float)
    demand = rng.integers(50, 150, size=n).astype(float)

    distance_supply = np.abs(points_supply).sum(axis=1).astype(float)
    distance_demand = np.abs(points_demand).sum(axis=1).astype(float)

    alpha = rng.uniform(1.2, 3.0, size=n)
    beta = rng.uniform(1.2, 3.0, size=n)
    lbd = alpha * distance_supply + 1.0
    gam = beta * distance_demand + 1.0

    return distance_supply, distance_demand, supply, demand, lbd, gam


def timed_solve(solver, data):
    start = time.perf_counter()
    solution = solver(*data)
    return solution, time.perf_counter() - start


def available_solvers():
    warmup = generate_dataset(20, [0, 20])
    active = {}

    for name, solver in SOLVERS.items():
        try:
            solver(*warmup)
            active[name] = solver
        except Exception as exc:
            print(f"Skip {name}: {exc}")

    return active


def benchmark(sizes, seeds, repeats=30):
    solvers = available_solvers()
    solver_names = list(solvers)
    records = []
    timing_records = []

    for n in sizes:
        for seed_index, seed in enumerate(seeds):
            data = generate_dataset(n, [int(seed), int(n)])
            distance_supply, distance_demand, supply, demand, lbd, gam = data
            beta_max = float(min(supply.sum(), demand.sum()))

            shift = seed_index % len(solver_names)
            run_order = solver_names[shift:] + solver_names[:shift]
            instance_results = {}

            for name in run_order:
                times = []
                solution = None

                try:
                    for repeat in range(1, repeats + 1):
                        solution, elapsed = timed_solve(solvers[name], data)
                        times.append(elapsed)
                        timing_records.append({
                            "n": n,
                            "seed": int(seed),
                            "repeat": repeat,
                            "solver": name,
                            "time_s": elapsed,
                        })
                except Exception as exc:
                    print(f"Skip {name} at n={n}, seed={seed}: {exc}")
                    continue

                u, v = solution
                obj = objective_value(
                    u, v,
                    distance_supply, distance_demand,
                    supply, demand, lbd, gam,
                )

                instance_results[name] = {
                    "time_s": float(np.median(times)),
                    "objective": obj,
                    "balance_error": abs(u.sum() - v.sum()),
                    "transported_mass": 0.5 * (u.sum() + v.sum()),
                }

            if REFERENCE not in instance_results:
                raise RuntimeError("KD-Sort did not finish.")

            reference_objective = instance_results[REFERENCE]["objective"]

            for name, info in instance_results.items():
                abs_error = abs(info["objective"] - reference_objective)
                rel_error = abs_error / max(1.0, abs(reference_objective))
                mass_error = abs(info["transported_mass"] - beta_max)

                records.append({
                    "n": n,
                    "seed": int(seed),
                    "solver": name,
                    "time_s": info["time_s"],
                    "objective": info["objective"],
                    "absolute_error_vs_KD": abs_error,
                    "relative_error_vs_KD": rel_error,
                    "balance_error": info["balance_error"],
                    "transported_mass": info["transported_mass"],
                    "mass_error_vs_beta_max": mass_error,
                })

        print(f"Finished n={n}")

    return pd.DataFrame(records), pd.DataFrame(timing_records)


def summarize_results(results):
    rows = []

    for (n, solver), subset in results.groupby(["n", "solver"], sort=True):
        rows.append({
            "n": n,
            "solver": solver,
            "successful_seeds": int(subset["time_s"].notna().sum()),
            "median_runtime_s": subset["time_s"].median(),
            "max_absolute_error_vs_KD": subset["absolute_error_vs_KD"].max(),
            "max_relative_error_vs_KD": subset["relative_error_vs_KD"].max(),
            "max_balance_error": subset["balance_error"].max(),
            "max_mass_error_vs_beta_max": subset["mass_error_vs_beta_max"].max(),
        })

    summary = pd.DataFrame(rows)
    kd_sort = summary[summary["solver"] == "KD-Sort"].set_index("n")["median_runtime_s"]
    kd_select = summary[summary["solver"] == "KD-Select"].set_index("n")["median_runtime_s"]

    summary["ratio_vs_KD_Sort"] = summary.apply(
        lambda row: row["median_runtime_s"] / kd_sort.loc[row["n"]], axis=1
    )
    summary["ratio_vs_KD_Select"] = summary.apply(
        lambda row: row["median_runtime_s"] / kd_select.loc[row["n"]], axis=1
    )

    return summary


def make_runtime_table(summary):
    runtime = summary.pivot(index="n", columns="solver", values="median_runtime_s")
    columns = [name for name in SOLVERS if name in runtime.columns]
    return runtime.reindex(columns=columns).reset_index()


def plot_results(results, summary):
    sizes = sorted(summary["n"].unique())
    solver_names = [name for name in SOLVERS if name in summary["solver"].unique()]

    plt.figure(figsize=(11, 7))
    for name in solver_names:
        subset = summary[summary["solver"] == name].sort_values("n")
        plt.plot(subset["n"], subset["median_runtime_s"], marker="o", label=name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Median runtime (seconds)")
    plt.title("UOT-W runtime comparison")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("runtime_medians_all_solvers.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(16, 8))
    group_width = 0.82
    box_width = group_width / len(solver_names)
    handles = []

    for j, name in enumerate(solver_names):
        data = []
        positions = []

        for i, n in enumerate(sizes):
            values = results[
                (results["n"] == n) & (results["solver"] == name)
            ]["time_s"].dropna().to_numpy()

            if values.size == 0:
                values = np.array([np.nan])

            data.append(values)
            positions.append(i + 1 - group_width / 2 + box_width / 2 + j * box_width)

        color = plt.cm.tab10(j % 10)
        box = plt.boxplot(
            data,
            positions=positions,
            widths=0.9 * box_width,
            patch_artist=True,
            manage_ticks=False,
        )

        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in box["medians"]:
            median.set_color("black")
        for whisker in box["whiskers"]:
            whisker.set_color(color)
        for cap in box["caps"]:
            cap.set_color(color)
        for flier in box["fliers"]:
            flier.set_markeredgecolor(color)

        handles.append(plt.Line2D([0], [0], color=color, lw=8, alpha=0.55))

    plt.xticks(range(1, len(sizes) + 1), [str(n) for n in sizes])
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime distributions of all solvers")
    plt.grid(True, axis="y")
    plt.legend(handles, solver_names, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("runtime_distributions_all_solvers.png", dpi=300, bbox_inches="tight")
    plt.close()


def print_environment():
    print("Python:", platform.python_version())
    print("NumPy:", np.__version__)
    print("SciPy:", scipy.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("CPU:", platform.processor() or platform.machine())
    print("Logical CPUs:", os.cpu_count())
    print("Solvers:", ", ".join(SOLVERS))


def main():
    sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    seeds = list(range(1, 31))
    repeats = 30

    print_environment()

    results, raw_times = benchmark(sizes=sizes, seeds=seeds, repeats=repeats)
    summary = summarize_results(results)
    runtime_table = make_runtime_table(summary)

    results.to_csv("benchmark_runs_all_solvers.csv", index=False)
    raw_times.to_csv("benchmark_raw_timings.csv", index=False)
    summary.to_csv("benchmark_summary_all_solvers.csv", index=False)
    runtime_table.to_csv("benchmark_runtime_table.csv", index=False)

    print("\nSummary")
    print(summary.to_string(index=False))

    plot_results(results, summary)


if __name__ == "__main__":
    main()
