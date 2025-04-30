from mip import (
    Model,
    xsum,
    minimize,
    BINARY,
    CONTINUOUS,
    MINIMIZE,
    CBC,
    GUROBI,
    ConstrsGenerator,
    CutType,
    OptimizationStatus,
    CutPool,
)
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import polars as pl

from time import time

import json
import sys


class CData:
    def __init__(self, instance_path):
        with open(instance_path, "r") as f:
            inst = json.load(f)

            self.N = inst["n"]
            self.demand_nodes = {}
            self.source_nodes = {}
            self.transhipment_nodes = []
            self.arcs = {}
            self.arcs_fixed_cost = {}
            self.node_requirement = {}
            for node in inst["nodes"]:
                self.node_requirement[node[0]] = node[1]
                if node[1] < 0:
                    self.source_nodes[node[0]] = node[1]
                elif node[1] > 0:
                    self.demand_nodes[node[0]] = node[1]
                else:

                    self.transhipment_nodes.append(node[0])

            for arc in inst["arcs"]:
                self.arcs_fixed_cost[(arc[0], arc[1])] = arc[3]
                self.arcs[(arc[0], arc[1])] = arc[2]

            # breakpoint()


class flow_model_improved:
    def __init__(self, dt):
        self.dt = dt
        self.arcs = dt.arcs
        self.arcs_fixed_cost = dt.arcs_fixed_cost
        self.demand_nodes = dt.demand_nodes
        self.source_nodes = dt.source_nodes
        self.N = dt.N
        self.create_model()

    def create_model(self):
        def variables(model):
            g_ijk = {
                (i, j, k): model.add_var(
                    "g({},{},{})".format(i, j, k),
                    var_type=CONTINUOUS,
                    obj=self.arcs[i, j],
                    lb=0.0,
                )
                for (i, j) in self.arcs
                for k in self.demand_nodes
            }
            q_jk = {
                (k, j): model.add_var(
                    "q({},{})".format(j, k), var_type=CONTINUOUS, lb=0.0
                )
                for k in self.demand_nodes
                for j in self.source_nodes
            }

            arc_activation = {
                (i, j): model.add_var(
                    "x({},{})".format(i, j),
                    var_type=BINARY,
                    obj=self.arcs_fixed_cost[i, j],
                )
                for (i, j) in self.arcs
            }

            return g_ijk, q_jk, arc_activation

        def constraints(model, g_ijk, q_jk, arc_activation):
            A = self.arcs
            # Eq. 26
            for j in self.source_nodes:
                for k in self.demand_nodes:
                    lhs = xsum(g_ijk[i, jj, k] for (i, jj) in A if jj == j) + q_jk[k, j]
                    rhs = xsum(g_ijk[jj, i, k] for (jj, i) in A if jj == j)
                    model += lhs == rhs, "eq_26_{},{}".format(j, k)

            # # Eq. 27
            for j in range(0, self.N):
                if j in self.source_nodes:
                    continue

                for k in self.demand_nodes:
                    if k == j:
                        continue

                    lhs = xsum(g_ijk[i, jj, k] for (i, jj) in A if jj == j)
                    rhs = xsum(g_ijk[jj, i, k] for (i, jj) in A if jj == j)
                    model += lhs == rhs, "eq_27_{},{}".format(j, k)

            # # Eq. 28
            for k, qty in self.demand_nodes.items():
                model += (
                    xsum(g_ijk[i, kk, k] for (i, kk) in A if kk == k)
                    - xsum(g_ijk[kk, j, k] for (kk, j) in A if kk == k)
                    == qty
                ), "eq_28_{}".format(k)

            # Eq. 29
            for k, qty in self.demand_nodes.items():
                for i, j in A:
                    model += g_ijk[i, j, k] <= qty * arc_activation[
                        i, j
                    ], "eq_29_{},({},{})".format(k, i, j)

            # Eq. 30
            for k, qty in self.demand_nodes.items():
                model += xsum(
                    q_jk[k, j] for j in self.source_nodes
                ) == qty, "eq_30_{}".format(k)

            # Eq. 31
            for j, qty in self.source_nodes.items():
                model += xsum(
                    q_jk[k, j] for k in self.demand_nodes
                ) == -qty, "eq_31_{}".format(j)

        model = Model(sense=MINIMIZE, solver_name=GUROBI)

        g_ijk, q_jk, arc_activation = variables(model)
        constraints(model, g_ijk, q_jk, arc_activation)

        self.model, self.arc_activation, self.g_ijk, self.q_jk = (
            model,
            arc_activation,
            g_ijk,
            q_jk,
        )

    def run(self, relax, verbose=False, show_status=False, lp_name=None):
        self.model.verbose = verbose
        self.model.optimize(relax=relax, max_seconds=600)
        # if lp_name is not None:
        #     self.model.write(lp_name)

        if show_status is True or True:
            print(f"-- Optimal objective {self.model.objective_value}")

        # print("\n".join([f"{k}, {x.x}" for k, x in self.arc_activation.items()]))
        arc_activation_values = {}
        for key, var in self.arc_activation.items():
            arc_activation_values[key] = var.x

        return arc_activation_values, self.model.objective_value

    def relax_and_cut(self):
        cut_counter = 0

        while True:
            arc_activation_values, _ = self.run(relax=True, verbose=False)

            dicut = dicut_model(self.dt, arc_activation_values, cut_counter)
            (obj_val, S, _S) = dicut.run()

            if abs(obj_val - 1.0) < 0.001:
                break

            arcs_cut = [(i, j) for (i, j) in self.arcs if i in _S and j in S]
            cuts = [self.arc_activation[i, j] for (i, j) in arcs_cut]
            # print(f"ADDED CUTS {arcs_cut}")

            self.model += xsum(cuts) >= 1, "cut{}".format(cut_counter)
            cut_counter += 1

        print(f"Added {cut_counter} cuts")
        self.run(relax=False, verbose=True)


class flow_model:
    def __init__(self, dt):
        self.dt = dt
        self.arcs = dt.arcs
        self.arcs_fixed_cost = dt.arcs_fixed_cost
        self.demand_nodes = dt.demand_nodes
        self.source_nodes = dt.source_nodes
        self.N = dt.N
        self.node_requirement = dt.node_requirement
        self.create_model()

    def create_model(self):
        def variables(model):
            arc_activation = {
                (i, j): model.add_var(
                    "x({},{})".format(i, j),
                    var_type=BINARY,
                    obj=self.arcs_fixed_cost[i, j],
                )
                for (i, j) in self.arcs
            }

            arc_flow = {
                (i, j): model.add_var(
                    "f({},{})".format(i, j), lb=0.0, obj=self.arcs[i, j]
                )
                for (i, j) in self.arcs
            }

            return arc_activation, arc_flow

        def constraints(model, arc_activation, arc_flow):
            for j in range(self.N):
                model += (
                    xsum(arc_flow[i, jj] for (i, jj) in self.arcs if jj == j)
                    - xsum(arc_flow[jj, i] for (jj, i) in self.arcs if jj == j)
                    == self.node_requirement[j]
                )

            big_D = sum(self.demand_nodes.values())
            for i, j in self.arcs:
                model += arc_flow[i, j] <= big_D * arc_activation[i, j]

        model = Model(sense=MINIMIZE, solver_name=GUROBI)

        arc_activation, arc_flow = variables(model)
        constraints(model, arc_activation, arc_flow)

        self.model, self.arc_activation, self.arc_flow = (
            model,
            arc_activation,
            arc_flow,
        )

    def run(self, relax, verbose=False, show_status=False, lp_name=None):
        self.model.verbose = verbose
        self.model.optimize(relax=relax, max_seconds=600)
        # if lp_name is not None:
        #     self.model.write(lp_name)

        if show_status is True or True:
            print(f"-- Optimal objective {self.model.objective_value}")

        # print("\n".join([f"{k}, {x.x}" for k, x in self.arc_activation.items()]))
        arc_activation_values = {}
        for key, var in self.arc_activation.items():
            arc_activation_values[key] = var.x

        return arc_activation_values, self.model.objective_value

    def relax_and_cut(self):
        cut_counter = 0

        while True:
            arc_activation_values, _ = self.run(relax=True, verbose=False)

            dicut = dicut_model(self.dt, arc_activation_values, cut_counter)
            (obj_val, S, _S) = dicut.run()

            if abs(obj_val - 1.0) < 0.001:
                break

            arcs_cut = [(i, j) for (i, j) in self.arcs if i in _S and j in S]
            cuts = [self.arc_activation[i, j] for (i, j) in arcs_cut]
            # print(f"ADDED CUTS {arcs_cut}")

            self.model += xsum(cuts) >= 1, "cut{}".format(cut_counter)
            cut_counter += 1

        print(f"Added {cut_counter} cuts")
        self.run(relax=False, verbose=True)


class dicut_model:
    def __init__(self, dt, arc_activation_values, added_cut):
        self.arcs = dt.arcs
        self.N = dt.N
        self.node_requirement = dt.node_requirement
        self.arc_activation_values = arc_activation_values
        self.added_cut = added_cut
        self.create_model()

    def create_model(self):
        def variables(model):
            w_var = {
                (i, j): model.add_var(
                    "w({},{})".format(i, j),
                    var_type=BINARY,
                )
                for (i, j) in self.arcs
            }

            z_var = {
                i: model.add_var("z({})".format(i), var_type=BINARY)
                for i in range(self.N)
            }

            return w_var, z_var

        def constraints(model, w_var, z_var):
            for i, j in self.arcs:
                model += w_var[i, j] <= z_var[j]
                model += w_var[i, j] <= (1 - z_var[i])
                model += w_var[i, j] >= z_var[j] + (1 - z_var[i]) - 1

            model += (
                xsum(self.node_requirement[i] * z_var[i] for i in range(self.N))
                >= 0.001
            )

        def objective_function(model, w_var):
            model.objective = minimize(
                xsum(
                    _x * w_var[i, j]
                    for (i, j), _x in self.arc_activation_values.items()
                )
            )

        model = Model(sense=MINIMIZE, solver_name=GUROBI)

        w_var, z_var = variables(model)
        constraints(model, w_var, z_var)
        objective_function(model, w_var)

        self.model, self.w_var, self.z_var = model, w_var, z_var

    def run(self):
        print("optimizing dicut model")
        self.model.verbose = False
        self.model.optimize(relax=False)

        S, _S = ([], [])

        for i in range(self.N):
            if self.z_var[i].x == 1:
                S.append(i)
            else:
                _S.append(i)

        assert self.model.objective_value >= 0.0
        return self.model.objective_value, S, _S


if __name__ == "__main__":
    print(f"running {sys.argv[1]}")
    dt = CData(sys.argv[1])
    improved = flow_model_improved(dt)
    # obj_imp = improved.run(relax=False, verbose=False, lp_name="improved.lp")

    # trivial = flow_model(dt)
    # obj_trivial = trivial.run(relax=False, verbose=False, lp_name="trivial.lp")

    stimer = time()
    improved.relax_and_cut()
    print(f"runtime: {time() - stimer:10.2f} s")
