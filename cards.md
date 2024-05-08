import random

def monte_carlo_draw_card(num_trials, colored_cards):
    occurrences = {color: 0 for color in colored_cards}

    for _ in range(num_trials):
        # Randomly choose a color from the colored cards
        selected_color = random.choice(colored_cards)
        # Increment the count of the chosen color
        occurrences[selected_color] += 1

    # Calculate probabilities
    probabilities = {color: count / num_trials for color, count in occurrences.items()}
    return probabilities

# Example usage
num_trials = 10000
colored_cards = ["red", "blue", "green", "yellow"]  # Example colors
probabilities = monte_carlo_draw_card(num_trials, colored_cards)

for color, probability in probabilities.items():
    print(f"Probability of drawing {color}: {probability:.4f}")