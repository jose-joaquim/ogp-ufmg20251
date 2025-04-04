import random
import polars as pl
from pyomo.environ import (
    Set,
    Param,
    RangeSet,
    Objective,
    Constraint,
    Var,
    ConcreteModel,
    Binary,
    NonNegativeIntegers,
    minimize,
    quicksum,
    SolverFactory,
    sum_product,
)


def basic_tsp(m: ConcreteModel, nodes: int, cost_matrix: dict[(int, int), int]):
    def sets_and_parameters():
        m.Nodes = RangeSet(1, nodes)
        m.Arcs = Set(initialize=m.Nodes * m.Nodes, dimen=2)
        m.ArcCost = Param(m.Arcs, initialize=cost_matrix, default=1)
        m.OriginNode = 1

        m.ArcsIntoOrigin = Set(initialize=[(u, v) for u, v in m.Arcs if v == 1])
        m.ArcsMinusArcsIntoOriginNode = m.Arcs - m.ArcsIntoOrigin

    def variables():
        m.ArcActivation = Var(m.Arcs, within=Binary)
        m.Order = Var(m.Nodes, within=NonNegativeIntegers)

    def constraints():
        m.leave_node = Constraint(
            m.Nodes,
            rule=lambda _, i: quicksum(
                m.ArcActivation[i, j] for (u, j) in m.Arcs if u == i
            )
            == 1,
        )

        m.enter_node = Constraint(
            m.Nodes,
            rule=lambda _, i: quicksum(
                m.ArcActivation[j, i] for (j, u) in m.Arcs if u == i
            )
            == 1,
        )

        m.sub_tour_elimination = Constraint(
            m.ArcsMinusArcsIntoOriginNode,
            rule=lambda _, i, j: m.Order[j]
            >= m.Order[i] + 1 - nodes * (1 - m.ArcActivation[i, j]),
        )

        m.subtours_degree_2 = Constraint(
            m.Arcs,
            rule=lambda _, i, j: m.ArcActivation[i, j] + m.ArcActivation[j, i] <= 1,
        )

    def objective():
        m.obj = Objective(expr=sum_product(m.ArcCost, m.ArcActivation), sense=minimize)

    sets_and_parameters()
    variables()
    constraints()
    objective()

    return m


def multicommodity_tsp(
    m: ConcreteModel, nodes: int, cost_matrix: dict[(int, int), int]
):
    def sets_and_parameters():
        m.Nodes = RangeSet(1, number_nodes)
        m.Arcs = Set(initialize=m.Nodes * m.Nodes, dimen=2)
        m.OriginNode = 1
        m.NodesButOrigin = m.Nodes - m.OriginNode

        m.CommodityDemand = Set()
        m.ArcDemand = Set()
        m.FixedCost = Param(m.Nodes, m.Nodes)

    def variables():
        m.ArcActivation = Var(m.Arcs, wthin=Binary)
        m.Flow = Var(m.Arcs, m.Demands, within=NonNegativeReals)
        m.VisitationOrder = Var(
            m.Nodes,
            within=Binary,
            bounds=[(0, 1) if u == m.OriginNode else (1, 1) for u in m.Nodes],
        )

    def constraints():
        m.flow_into = Constraint(
            m.Nodes, lambda _, i: quicksum(m.Flow[i, j] for (i, j) in m.Arcs) == 1
        )
        m.flow_out = Constraint(
            m.Nodes, lambda _, j: quicksum(m.Flow[i, j] for i in m.Arcs) == 1
        )

        m.flow_arc_coupling = Constraint(
            m.Arcs,
            m.Demands,
            rule=lambda _, i, j, k: m.Flow[i, j, k] <= m.ArcActivation[i, j],
        )

        m.conservation_1 = Constraint(
            expr=quicksum(m.Flow[m.OriginNode, j, m.OriginNode] for j in m.Nodes) == 1
        )

        m.conservation_2 = Constraint(
            expr=quicksum(m.Flow[j, m.OriginNode, m.OriginNode] for j in m.Nodes) == 1
        )

        m.conservation_3 = Constraint(
            m.ArcDemand,
            rule=lambda _, k: quicksum(m.Flow[m.OriginNode, j, k] for j in m.Nodes)
            - quicksum(m.Flow[j, m.OriginNode, k] for j in m.Nodes)
            == 1,
        )

        m.conservation_4 = Constraint(
            m.ArcDemand,
            rule=lambda _, k: quicksum(m.Flow[i, k, k] for i in m.Nodes if i != k)
            - quicksum(m.Flow[k, i, k] for i in m.Nodes if i != k)
            == 1,
        )

        m.conservation_5 = Constraint(
            m.Nodes,
            m.ArcDemand,
            rule=lambda _, j, k: quicksum(
                m.Flow[j, i, k] for i in m.Nodes if i not in [j, k, m.OriginNode]
            )
            - quicksum(
                m.Flow[i, j, k] for i in m.Nodes if i not in [j, k, m.OriginNode]
            )
            == 0,
        )

        m.assignment_node_order1 = Constraint(
            m.Order, rule=lambda _, w: quicksum(m.P[i, w] for i in m.Nodes) == 1
        )

        m.assignment_node_order2 = Constraint(
            m.Order, rule=lambda _, i: quicksum(m.P[i, w] for w in m.Nodes) == 1
        )

    def objective():
        m.obj = Objective(
            expr=sum_product(m.FixedCost, m.Arcs)
            + quicksum(
                m.FlowCost[i, j, k] * m.CommodityDemand[k] + m.Flow[i, j, k]
                for i, j, k in m.ArcDemand
            ),
            sense=minimize,
        )


solver = SolverFactory("gurobi_direct")
nodes_set = pl.int_range(10, 101, eager=True).sample(n=10, seed=1).to_list()
for number_nodes in nodes_set:
    number_nodes = 13
    cost_matrix = {
        (i + 1, j + 1): int(100 * random.random())
        for i in range(number_nodes)
        for j in range(number_nodes)
    }

    m = ConcreteModel()
    basic_tsp(m, number_nodes, cost_matrix)

    result = solver.solve(m, tee=True, symbolic_solver_labels=True)
    solver._solver_model.write("sol.sol")
    for v in [x for x in m.component_data_objects(Var) if x.value >= 1]:
        print(str(v), v.value)

    breakpoint()
