from pyomo.environ import (
    Set,
    Param,
    RangeSet,
    Objective,
    Constraint,
    Var,
    ConcreteModel,
    Binary,
    NonNegativeReals,
    minimize,
    quicksum,
    SolverFactory,
    sum_product,
)


def sets_and_parameters(m):
    m.N = 10
    m.CommodityDemand = Set()
    m.Nodes = RangeSet(1, m.N)
    m.Arcs = Set(m.Nodes, m.Nodes)
    m.ArcDemand = Set()
    m.FixedCost = Param(m.Nodes, m.Nodes)
    m.OriginNode = 1


def variables(m):
    m.ArcActivation = Var(m.Arcs, wthin=Binary)
    m.Flow = Var(m.Arcs, m.Demands, within=NonNegativeReals)
    m.VisitationOrder = Var(
        m.Nodes,
        within=Binary,
        bounds=[(0, 1) if u == m.OriginNode else (1, 1) for u in m.Nodes],
    )


def constraints(m):
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
        - quicksum(m.Flow[i, j, k] for i in m.Nodes if i not in [j, k, m.OriginNode])
        == 0,
    )

    m.assignment_node_order1 = Constraint(
        m.Order, rule=lambda _, w: quicksum(m.P[i, w] for i in m.Nodes) == 1
    )

    m.assignment_node_order2 = Constraint(
        m.Order, rule=lambda _, i: quicksum(m.P[i, w] for w in m.Nodes) == 1
    )


def objective(m):
    m.obj = Objective(
        expr=sum_product(m.FixedCost, m.Arcs)
        + quicksum(
            m.FlowCost[i, j, k] * m.CommodityDemand[k] + m.Flow[i, j, k]
            for i, j, k in m.ArcDemand
        ),
        sense=minimize,
    )


m = ConcreteModel()

sets_and_parameters(m)
variables(m)
constraints(m)
objective(m)

solver = SolverFactory("gurobi")
result = solver.solve(m, tee=True)
