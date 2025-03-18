from dimod import BinaryQuadraticModel
import numpy as np
import neal

import matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def build_bqm(num_pumps, time, power, costs, flow, demand, v_init, v_min, v_max, c3_gamma):
    """
    Build a binary quadratic model (BQM) for the pump scheduling problem.

    Args:
        num_pumps (int): Number of pumps
        time (list of int): Time slots
        power (list of floats): power[p] = power required for pump p
        costs (list of floats): costs[t] = cost of power at time t
        flow (list of floats): flow[p] = water flow from pump p
        demand (list of floats): demand[t] = outflow at time t
        v_init (float): initial reservoir level
        v_min (float): minimum reservoir level
        v_max (float): maximum reservoir level
        c3_gamma (float): Lagrange multiplier for water-level constraint

    Returns:
        bqm (BinaryQuadraticModel): The built BQM
        x (list of list): x[p][t] - variable names for pump p at time t
    """

    print("\nBuilding binary quadratic model...")

    # Create variable names x[p][t]
    x = [
        [f"P{p}_{t}" for t in time]
        for p in range(num_pumps)
    ]

    # Initialize BQM as a BinaryQuadraticModel
    bqm = BinaryQuadraticModel("BINARY")

    # Objective: minimize total cost
    # total_cost = gamma * sum_{p,t} (power[p]*costs[t]/1000)* x[p][t]
    gamma = 10000.0
    for p in range(num_pumps):
        for t_index, t_val in enumerate(time):
            # Force float to avoid int8 overflow
            cost_contribution = float(gamma)*float(power[p])*float(costs[t_index]) / 1000.0
            bqm.add_variable(x[p][t_index], cost_contribution)

    # Constraint 1: Each pump runs at least once per day
    # sum_{t} x[p][t] >= 1 => lb=1, ub=len(time)
    for p in range(num_pumps):
        c1 = [(x[p][t], 1) for t in range(len(time))]
        bqm.add_linear_inequality_constraint(
            c1,
            lb=1,
            ub=len(time),
            lagrange_multiplier=1,
            label=f"c1_pump_{p}"
        )

    # Constraint 2: At most (num_pumps - 1) pumps can run simultaneously
    # sum_{p} x[p][t] <= num_pumps - 1
    for t_index, t_val in enumerate(time):
        c2 = [(x[p][t_index], 1) for p in range(num_pumps)]
        bqm.add_linear_inequality_constraint(
            c2,
            constant=- (num_pumps - 1),
            lagrange_multiplier=1,
            label=f"c2_time_{t_val}"
        )

    # Constraint 3: Water level between v_min and v_max at each time
    # reservoir[t] = v_init + sum_{k=0..t} sum_{p} flow[p]* x[p][k] - sum_{k=0..t} demand[k]
    # multiply flow by 100 for integer
    for t_index, t_val in enumerate(time):
        c3 = []
        for p in range(num_pumps):
            # For each time <= t_index
            for k in range(t_index+1):
                c3.append((x[p][k], int(flow[p]*100)))  # int
        # initial reservoir *100
        const = int(v_init*100)
        # subtract demand up to t_index
        const -= int(sum(demand[:t_index+1]) * 100)

        lb_ = int(v_min*100)
        ub_ = int(v_max*100)

        bqm.add_linear_inequality_constraint(
            c3,
            constant=const,
            lb=lb_,
            ub=ub_,
            lagrange_multiplier=c3_gamma,
            label=f"c3_time_{t_val}"
        )

    return bqm, x

def process_sample(sample, x, pumps, time, power, flow, costs, demand, v_init, verbose=True):
    """
    Process the best sample (var->0/1) from the solver and compute:
      - total_flow, total_cost
      - reservoir water levels
    """

    print("\nProcessing sampleset returned...\n")

    total_flow = 0.0
    total_cost = 0.0
    num_pumps = len(pumps)

    if verbose:
        # Print time header
        timeslots = "\n\t" + "\t".join(str(t) for t in time)
        print(timeslots)

    # Print usage per pump
    for p in range(num_pumps):
        line_out = str(pumps[p])
        for t_index, t_val in enumerate(time):
            val = float(sample[x[p][t_index]])  # cast to float
            line_out += f"\t{val}"
            total_flow += val * float(flow[p])
            total_cost += val * float(costs[t_index]) * float(power[p]) / 1000.0
        if verbose:
            print(line_out)

    # Compute reservoir levels
    reservoir = [v_init]
    pump_flow_schedule = []
    if verbose:
        level_str = "Level:\t"
    for t_index, t_val in enumerate(time):
        inflow = 0.0
        for p in range(num_pumps):
            inflow += float(sample[x[p][t_index]]) * float(flow[p])
        new_level = reservoir[-1] + inflow - demand[t_index]
        reservoir.append(new_level)
        pump_flow_schedule.append(inflow)
        if verbose:
            level_str += str(int(new_level)) + "\t"
    if verbose:
        print("\n" + level_str)

    print("\nTotal flow:\t", total_flow)
    print("Total cost:\t", total_cost, "\n")

    return pump_flow_schedule, reservoir

def visualize(sample, x, v_min, v_max, v_init, num_pumps, costs, power, pump_flow_schedule, reservoir, time):
    """
    Build an animation of the reservoir water level, saving to reservoir.html
    """

    print("\nBuilding visualization...")

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, v_min/2 + v_max)
    ax.xaxis.set_visible(False)
    ax.set_yticks([v_min, v_max])
    ax.set_yticklabels(('','Max'))
    ax.set_title("Reservoir Water Level")

    ax.plot(range(5), [v_max]*5, color='#222222', label="Max capacity", linewidth=1.0)
    ax.plot(range(5), [v_min]*5, color='#FFA143', label="Min capacity", linewidth=1.5)

    barcollection = plt.bar(0.5, v_init, width=1.0, color='#2a7de1', align='center')
    water_line, = ax.plot([], [], 'b-')
    x_ax_vals = np.linspace(0, 1, 200)

    pumps_used = []
    for i in range(num_pumps):
        pumps_used.append(plt.figtext(0.03, 0.11+0.035*i, f"Pump {i+1}", color='#DDDDDD', fontsize='small'))
    time_label = ax.text(0.75, 1600, '')
    cost_label = plt.figtext(0.45, 0.03, '', color='k')

    def animate(frame):
        smoothing_factor = 4
        m = frame % (60/smoothing_factor)
        t = int((frame - m)/(60/smoothing_factor))

        pump_min_flow = m*smoothing_factor * pump_flow_schedule[t]/60.0
        demand_min = m*smoothing_factor * demand[t]/60.0
        delta = reservoir[t] + pump_min_flow - demand_min
        y = [delta]*len(x_ax_vals)
        for b in barcollection:
            b.set_height(delta)
        water_line.set_data(x_ax_vals, y)

        time_label.set_text(f"Time: {time[t]}")
        cost = 0.0
        for p in range(num_pumps):
            if float(sample[x[p][t]]) == 1.0:
                pumps_used[p].set_color('#008c82')
                cost += float(costs[t])*float(power[p]) / 1000.0
            else:
                pumps_used[p].set_color('#DDDDDD')
        cost_label.set_text(f"Hourly Cost: {cost}")
        return water_line,

    from matplotlib import animation
    smoothing_factor = 4
    frames = int(24*(60/smoothing_factor))
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=2, blit=True)
    mywriter = animation.HTMLWriter(fps=30)
    anim.save('reservoir.html', writer=mywriter)

    print("\nAnimation saved as reservoir.html.")

if __name__ == '__main__':

    # Problem parameters
    num_pumps = 7
    pumps = [f'P{i+1}' for i in range(num_pumps)]
    time = list(range(1, 25))
    power = [15, 37, 33, 33, 22, 33, 22]
    costs = [169]*7 + [283]*6 + [169]*3 + [336]*5 + [169]*3
    flow = [75, 133, 157, 176, 59, 69, 120]
    demand = [
        44.62, 31.27, 26.22, 27.51, 31.50, 46.18, 69.47, 100.36,
        131.85, 148.51, 149.89, 142.21, 132.09, 129.29, 124.06,
        114.68, 109.33, 115.76, 126.95, 131.48, 138.86, 131.91,
        111.53, 70.43
    ]
    v_init = 550.0
    v_min = 523.5
    v_max = 1500.0
    c3_gamma = 0.00052

    # 1) Build BQM
    bqm, x = build_bqm(num_pumps, time, power, costs, flow, demand,
                       v_init, v_min, v_max, c3_gamma)

    # 2) Solve BQM with classical simulated annealing
    print("\nRunning classical solver (neal.SimulatedAnnealingSampler)...")
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=1000)
    sample = sampleset.first.sample

    # 3) Process best solution
    pump_flow_schedule, reservoir = process_sample(
        sample, x, pumps, time, power, flow, costs, demand, v_init
    )

    # 4) Visualize
    visualize(sample, x, v_min, v_max, v_init, num_pumps, costs, power,
              pump_flow_schedule, reservoir, time)
