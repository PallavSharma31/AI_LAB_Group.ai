import numpy as np
from scipy.stats import norm, poisson

# Define the population distribution by age group
pop_age_groups = np.array([0.38, 0.33, 0.23, 0.06])
pop_total = 8700000  # Total population of the city

# Define the initial number of normal and covid beds
beds_normal_max = 650
beds_covid_max = 0
discount_factor=0.9
# Define the cost parameters
cost_bed_normal_denied = 500
cost_bed_covid_denied = 2000
cost_bed_convert = 10000

# Define the range of beds to convert from normal to covid
beds_convert_range = np.arange(0, beds_normal_max + 1, 50)

# Define the transition function
def get_transition_prob(state, action):
    # Unpack the state
    pop_age_groups, pop_total, beds_normal, beds_covid = state
    
    # Calculate the number of requests for normal and covid beds
    requests_normal = np.sum(poisson.rvs(0.1, size=len(pop_age_groups)) * pop_age_groups)
    requests_covid = np.sum(poisson.rvs(0.05, size=len(pop_age_groups)) * pop_age_groups)
    
    # Calculate the number of discharges for normal and covid beds
    discharges_normal = poisson.rvs(beds_normal)
    discharges_covid = poisson.rvs(beds_covid,)
    
    # Calculate the number of denied requests for normal and covid beds
    denied_normal = max(requests_normal - beds_normal - action, 0)
    denied_covid = max(requests_covid - beds_covid, 0)
    
    # Calculate the new state after accounting for the action and the probabilistic transitions
    new_beds_normal = max(min(beds_normal + requests_normal - discharges_normal - action, beds_normal_max), 0)
    new_beds_covid = max(min(beds_covid + requests_covid - discharges_covid, beds_normal_max), 0)
    new_pop_age_groups = pop_age_groups + norm.rvs(scale=1, size=len(pop_age_groups))
    new_pop_total = max(min(pop_total + norm.rvs(scale=0.1), 10000000), 0)
    
    # Calculate the reward for the transition
    reward = -cost_bed_normal_denied * denied_normal \
             -cost_bed_covid_denied * denied_covid \
             -cost_bed_convert * action
    
    return ((new_pop_age_groups, new_pop_total, new_beds_normal, new_beds_covid), reward)

# Define the value iteration function to solve the MDP
def value_iteration(theta=0.0001, discount_factor=0.9):
    V = np.zeros((pop_age_groups.size+2, beds_normal_max+1, 4))
    delta = float('inf')
    while delta > theta:
        delta = 0
        for age_group in range(pop_age_groups.size+2):
            for beds_normal in range(beds_normal_max+1):
                for week in range(4):
                    v = V[age_group, beds_normal, week]
                    action_values = []
                    for action in beds_convert_range:
                        next_state, reward = get_transition_prob((pop_age_groups, pop_total, beds_normal, beds_covid_max), action)
                        next_age_group = np.clip(np.sum(next_state[0] * pop_age_groups / next_state[1]), 0                        , len(pop_age_groups)+1)
                        next_beds_normal = next_state[2]
                        next_week = week + 1 if week < 3 else 0
                        action_values.append(reward + discount_factor * V[next_age_group, next_beds_normal, next_week])
                    V[age_group, beds_normal, week] = max(action_values)
                    delta = max(delta, abs(v - V[age_group, beds_normal, week]))
    return V

# Solve the MDP
V = value_iteration()

# Extract the optimal policy
policy = np.zeros((pop_age_groups.size+2, beds_normal_max+1, 4))
for age_group in range(pop_age_groups.size+2):
    for beds_normal in range(beds_normal_max+1):
        for week in range(4):
            action_values = []
            for action in beds_convert_range:
                next_state, reward = get_transition_prob((pop_age_groups, pop_total, beds_normal, beds_covid_max), action)
                next_age_group = np.clip(np.sum(next_state[0] * pop_age_groups / next_state[1]), 0, len(pop_age_groups)+1)
                next_beds_normal = next_state[2]
                next_week = week + 1 if week < 3 else 0
                action_values.append(reward + discount_factor * V[next_age_group, next_beds_normal, next_week])
            policy[age_group, beds_normal, week] = beds_convert_range[np.argmax(action_values)]

# Print the optimal policy for each week
for week in range(4):
    print(f"Week {week+1}:")
    print("Age Group\tNormal Beds\tConvert to Covid Beds")
    for age_group in range(pop_age_groups.size+2):
        for beds_normal in range(beds_normal_max+1):
            print(f"{age_group}\t\t{beds_normal}\t\t{policy[age_group, beds_normal, week]}")