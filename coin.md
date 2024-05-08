# Import the random module
import random

a = random.uniform(0, 1)

if a>0.5:
    print("Head")
else:
    print("Tail")

def unbiased_coin_toss():
    # Generate a random number from 0 to 1
    x = random.uniform(0, 1)
    # Probability that the number falls between 0 and 0.5 is 1/2

    if x > 0.5:
        # Heads for True
        return True
    else:
        # Tails for False
        return False

for i in range(10):
    print(unbiased_coin_toss())

N=10

# List to store the result(True/False)
results = []

# Toss the coin 10 times and store that in a list
for i in range(N):
    result = unbiased_coin_toss()
    results.append(result)

# Find the total number of heads
n_heads = sum(results)

# Find probability of heads in the experiment
p_heads = n_heads/N

print("Probability is {:.1f}".format(p_heads))


# Monte Carlo Simulation for toss
prob = []

# Make 1000 experiments
for i in range(1000):

    # Each experiment have 10 coin tosses
    N = 10
    results = []

    # Toss the coin 10 times and store that in a list
    for i in range(N):
        result = unbiased_coin_toss()
        results.append(result)

    n_heads = sum(results)
    p_heads = n_heads/N
    prob.append(p_heads)

# average the probability of heads over 1000 experiments
p_heads_MC = sum(prob)/1000

print("Probability is {:.3f}".format(p_heads_MC))