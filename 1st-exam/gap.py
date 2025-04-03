from pyomo.environ import (
    Set,
    Param,
    RangeSet,
    Objective,
    Constraint,
    Var,
    ConcreteModel,
    Binary,
    minimize,
    maximize,
    quicksum,
    summation,
    SolverFactory,
    SolverStatus,
    sum_product,
)

v = {("cut", "money"): 1007}

m = ConcreteModel()

m.tasks = Set(initialize=["cut", "clean", "run", "sleep", "eat"])
m.resources = Set(initialize=["money", "food", "car", "heat"])
m.Benefit = Param(m.tasks, m.resources, default=1, initialize=v)
m.Consumption = Param(m.tasks, m.resources, default=1, initialize=v)
m.Priority = Param(m.tasks, default=10000007)
m.Capacity = Param(m.resources, default=1)

m.Allocation = Var(m.tasks, m.resources, within=Binary)
m.Active = Var(m.tasks, within=Binary)

m.coupling = Constraint(
    m.tasks,
    rule=lambda _, i: quicksum(m.Allocation[i, j] for j in m.resources) >= m.Active[i],
)

m.summing = Constraint(
    m.resources,
    rule=lambda _, j: quicksum(
        m.Consumption[i, j] * m.Allocation[i, j] for i in m.tasks
    )
    <= m.Capacity[j],
)

m.cost = Objective(
    expr=sum_product(m.Benefit, m.Allocation) - sum_product(m.Priority, m.Active),
    sense=minimize,
)


m.name = "The GAP problem with big-M"
m.pprint()

solver = SolverFactory("gurobi")
m.write("model.lp", io_options={"symbolic_solver_labels": True})
ans = solver.solve(m, tee=True, symbolic_solver_labels=True)

if ans.solver.status == SolverStatus.ok:
    print("Print values for all variables")
    for v in m.component_data_objects(Var):
        print(str(v), v.value)
