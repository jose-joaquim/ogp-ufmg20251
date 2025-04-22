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
from networkx import DiGraph
import networkx as nx

import polars as pl

from time import time
import numpy as np


class CData:
    def __init__(self, n):
        assert n > 4, "number of nodes must be greater than 4"
        self.n = n
        self.coord = np.random.uniform(0, 100, size=(n, 2))
        self.c = squareform(pdist(self.coord)).astype(int)
        self.N = range(n)
        self.A = [(i, j) for i in self.N for j in self.N if i != j]
        self.F = sorted(
            [(i, j, self.c[i][j]) for (i, j) in self.A],
            key=lambda t: t[2],
            reverse=True,
        )


class CMTZ:
    def __init__(self, dt):
        self.dt = dt
        self.create_model()

    def create_model(self):
        dt = self.dt
        A, N, n, c = dt.A, dt.N, dt.n, dt.c
        model = Model(sense=MINIMIZE, solver_name=GUROBI)
        x = {
            (i, j): model.add_var("x({},{})".format(i, j), var_type=BINARY, obj=c[i][j])
            for (i, j) in A
        }
        t = {
            i: model.add_var("t({})".format(i), var_type=CONTINUOUS, lb=0.0) for i in N
        }
        for i in N:
            model += xsum(x[(ii, j)] for (ii, j) in A if i == ii) == 1
        for j in N:
            model += xsum(x[(i, jj)] for (i, jj) in A if j == jj) == 1
        for i, j in A:
            if j != 0:
                model += t[i] - t[j] + n * x[i, j] <= n - 1
        self.model, self.x, self.t = model, x, t

    def run(self, relax=False, verbose=0, show_status=False, lp_name=None):
        dt = self.dt
        N, A = dt.N, dt.A
        model, t, x = self.model, self.t, self.x
        model.verbose = verbose
        if lp_name is not None:
            model.write(lp_name)
        model.optimize(relax=relax, max_seconds=600)
        assert model.num_solutions >= 1, "Error solving model MTZ"
        self.obj_val = model.objective_value
        _t = [t[i].x for i in N]
        _x = {(i, j): x[i, j].x for (i, j) in A if x[i, j].x > 1e-6}
        self._x, self._t = _x, _t

        if show_status is True:
            print(f"-- Optimal objective {model.objective_value}")

    def hot_start(self, hmax=10):
        dt = self.dt
        A, F = dt.A, dt.F
        model, x = self.model, self.x
        h = 0
        stop = False
        stimer = time()
        added_cuts = 0
        while stop is False:
            h = h + 1
            self.run(relax=True)
            # separating subtour cuts
            G = DiGraph()
            for i, j in A:
                G.add_edge(i, j, capacity=self.x[i, j].x)
            stop = True
            for i, j, val in F:
                if j > 0:
                    val, (S, _S) = nx.minimum_cut(G, i, j)
                    if val <= 1 - 1e-6:
                        added_cuts += 1
                        model += (
                            xsum(x[i, j] for (i, j) in A if i in S and j in S)
                            <= len(S) - 1
                        )
                        stop = False
            if h == hmax:
                stop = True
            print(f" {h:3d} {self.obj_val:10.2f} {time() - stimer:10.2f} s")

        print(f"Added {added_cuts} cuts")


class GG:
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

            edge_flow = {(i, j): model.add_var("f({},{})".format(i, j)) for (i, j) in A}

            return edge_activation, edge_flow

        def constraints(model, edge_activation, edge_flow, A, n):
            for i in range(n):
                model += xsum(edge_activation[(ii, j)] for (ii, j) in A if i == ii) == 1

            for j in range(n):
                model += xsum(edge_activation[(i, jj)] for (i, jj) in A if j == jj) == 1

            model += xsum(edge_flow[i, j] for (i, j) in A if i == 0) == n - 1

            for j in range(1, n):
                model += (
                    xsum(edge_flow[i, jj] for (i, jj) in A if j == jj)
                    - xsum(edge_flow[jj, i] for (jj, i) in A if j == jj)
                    == 1
                )

            for i, j in A:
                model += edge_flow[i, j] <= (n - 1) * edge_activation[i, j]

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

    def hot_start(self, hmax=10, verbose=False):
        h = 0
        dt = self.dt
        A, F = dt.A, dt.F
        model, edge_activation = self.model, self.edge_activation
        stop = False
        added_cuts = 0
        stimer = time()
        while stop is False:
            h = h + 1
            self.run(relax=True, verbose=verbose)

            G = DiGraph()
            for i, j in A:
                G.add_edge(i, j, capacity=self.edge_activation[i, j].x)

            stop = True
            for i, j, val in F:
                if j > 0:
                    val, (S, _S) = nx.minimum_cut(G, i, j)
                    if val <= 1 - 1e-6:
                        added_cuts += 1
                        model += (
                            xsum(
                                edge_activation[i, j]
                                for (i, j) in A
                                if i in S and j in S
                            )
                            <= len(S) - 1
                        )
                        stop = False

            if h == hmax:
                stop = True

            print(
                f" {h:3d} {self.model.objective_value:10.2f} {time() - stimer:10.2f} s"
            )

        print(f"Added {added_cuts} cuts")


if __name__ == "__main__":
    seed = 10
    shuffled_instances = (
        pl.int_range(10, 101, eager=True).sample(5, seed=20).sort().to_list()
    )

    print(f"Instances are {shuffled_instances}")
    for nodes in shuffled_instances:
        dt = CData(nodes)

        print(f"Running GG with {nodes} nodes")
        gg = GG(dt)
        gg.run(
            relax=False, show_status=True, verbose=False, lp_name=f"gg{nodes}_raw.lp"
        )

        gg = GG(dt)
        gg.hot_start(10, verbose=False)
        gg.run(
            relax=False, show_status=True, verbose=False, lp_name=f"gg{nodes}_cut.lp"
        )

        ##############################################

        print(f"Running MTZ with {nodes} nodes")
        mtzcp = CMTZ(dt)
        mtzcp.run(
            relax=False, show_status=True, verbose=False, lp_name=f"mtz{nodes}_raw.lp"
        )

        mtzcp = CMTZ(dt)
        mtzcp.hot_start(10)
        mtzcp.run(
            relax=False, show_status=True, verbose=False, lp_name=f"mtz{nodes}_cut.lp"
        )
        print("end\n\n\n")
        breakpoint()
