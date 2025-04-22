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


class CData:
    def __init__(self):
        with open("./n10-a36-d5-s3-py.dat", "r") as f:
            inst = json.load(f)

            self.N = inst["n"]
            self.demand_nodes = {}
            self.source_nodes = {}
            self.transhipment_nodes = []
            self.arcs = {}
            self.arcs_fixed_cost = {}
            for node in inst["nodes"]:
                if node[1] < 0:
                    self.source_nodes[node[0]] = node[1]
                elif node[1] > 0:
                    self.demand_nodes[node[0]] = node[1]
                else:
                    self.transhipment_nodes.append(node[0])

            for arc in inst["arcs"]:
                self.arcs_fixed_cost[(arc[0], arc[1])] = arc[2]
                self.arcs[(arc[0], arc[1])] = arc[3]


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

        model.write("improved.lp")

        self.model, self.arc_activation, self.g_ijk, self.q_jk = (
            model,
            arc_activation,
            g_ijk,
            q_jk,
        )


class flow_model:
    def __init__(self, dt):
        self.dt = dt
        self.create_model()

    def create_model(self):
        def variables(model, A, c):
            edge_activation = {
                (i, j): model.add_var(
                    "x({},{})".format(i, j), var_type=BINARY, obj=c[i][j]
                )
                for (i, j) in A
            }

            edge_flow = {
                (i, j): model.add_var("f({},{})".format(i, j), lb=0.0) for (i, j) in A
            }

            return edge_activation, edge_flow

        def constraints(
            model, edge_activation, edge_flow, A, n, flow_req, demand_nodes
        ):
            for j in range(10):
                model += xsum() - xsum() == flow_req[j]

            for i, j in A:
                model += (
                    edge_flow[i, j]
                    <= max(demand_nodes.values()) * edge_activation[i, j]
                )

        dt = self.dt
        A, n, c = dt.A, dt.n, dt.c
        model = Model(sense=MINIMIZE, solver_name=GUROBI)

        edge_activation, edge_flow = variables(model, A, c)
        constraints(model, edge_activation, edge_flow, A, n)

        self.model, self.edge_activation, self.edge_flow = (
            model,
            edge_activation,
            edge_flow,
        )

    def run(self, relax, verbose=False, show_status=False, lp_name=None):
        self.model.verbose = verbose
        self.model.optimize(relax=relax, max_seconds=600)
        if lp_name is not None:
            self.model.write(lp_name)

        if show_status is True:
            print(f"-- Optimal objective {self.model.objective_value}")


if __name__ == "__main__":
    seed = 10
    dt = CData()
    improved = flow_model_improved(dt)
