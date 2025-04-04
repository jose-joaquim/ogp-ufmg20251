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
import json
import random

with open("./gap-instances.json", "r") as f:
    instances = json.load(f)

for instance in instances:
    m = ConcreteModel()

    # 1. Sets
    m.Tasks = RangeSet(1, instance["ni"])
    m.Resources = RangeSet(1, instance["nj"])

    # 2. Parameters
    m.AllocationCost = Param(
        m.Tasks, m.Resources, initialize=lambda _, i, j: instance["c"][i - 1][j - 1]
    )
    m.Consumption = Param(
        m.Tasks, m.Resources, initialize=lambda _, i, j: instance["t"][i - 1][j - 1]
    )
    m.Capacity = Param(m.Resources, initialize=lambda _, _2: int(100 * random.random()))
    m.Benefit = Param(m.Tasks, initialize=lambda _, _2: int(100 * random.random()))

    # 3. Variables
    m.Allocation = Var(m.Tasks, m.Resources, within=Binary)
    m.Active = Var(m.Tasks, within=Binary)

    # 4. Constraints
    m.coupling = Constraint(
        m.Tasks,
        rule=lambda _, i: quicksum(m.Allocation[i, j] for j in m.Resources)
        >= m.Active[i],
    )

    m.summing = Constraint(
        m.Resources,
        rule=lambda _, j: quicksum(
            m.Consumption[i, j] * m.Allocation[i, j] for i in m.Tasks
        )
        <= m.Capacity[j],
    )

    # 5. Objective Function
    m.cost = Objective(
        expr=sum_product(m.AllocationCost, m.Allocation)
        - sum_product(m.Benefit, m.Active),
        sense=minimize,
    )

    m.name = "The GAP problem with big-M"
    solver = SolverFactory("gurobi_direct")

    ans = solver.solve(m, tee=True, symbolic_solver_labels=True)
    if ans.solver.status == SolverStatus.ok:
        ans.write()
        print("\n\nPrint values for all variables")
        for v in m.component_data_objects(Var):
            print(str(v), v.value)
