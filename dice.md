import random

def monte_carlo_dice_roll(num_trials):
    occurrences = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    for _ in range(num_trials):
        # Randomly choose a number between 1 and 6 (inclusive)
        outcome = random.randint(1, 6)
        # Increment the count of the chosen outcome
        occurrences[outcome] += 1

    # Calculate probabilities
    probabilities = {number: count / num_trials for number, count in occurrences.items()}
    return probabilities

# Example usage
num_trials = 10000
probabilities = monte_carlo_dice_roll(num_trials)

for number, probability in probabilities.items():
    print(f"Probability of getting {number}: {probability:.4f}")

#Biased dice
import random

def monte_carlo_biased_dice_roll(num_trials, probabilities):
    occurrences = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    for _ in range(num_trials):
        # Randomly choose a number based on custom probabilities
        outcome = random.choices(list(occurrences.keys()), weights=probabilities)[0]
        # Increment the count of the chosen outcome
        occurrences[outcome] += 1

    # Calculate probabilities
    probabilities = {number: count / num_trials for number, count in occurrences.items()}
    return probabilities

# Example usage
num_trials = 10000
custom_probabilities = {1: 0.2, 2: 0.1, 3: 0.15, 4: 0.15, 5: 0.2, 6: 0.2}
probabilities = monte_carlo_biased_dice_roll(num_trials, custom_probabilities)

for number, probability in probabilities.items():
    print(f"Probability of getting {number}: {probability:.4f}")

