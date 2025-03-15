
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            !pip install prompt_gen
!pip install -upgrade pip

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import random
import matplotlib.pyplot as plt

# Recursive function for text expansion
def recursive_text_expansion(seed, iterations):
    results = [seed]

    for i in range(iterations):
        # Simulate variation and context coherence
        variation = random.choice(["increase", "decrease", "maintain"])
        context = results[-1]

        if variation == "increase":
            new_text = context + " more information"
        elif variation == "decrease":
            new_text = context[:-len(" information")]
        else:
            new_text = context

        results.append(new_text)

    return results

# Initial seed
seed_text = "This is an AI model that generates"

# Number of iterations
num_iterations = 5

# Generate text recursively
generated_text = recursive_text_expansion(seed_text, num_iterations)

# Visualize the generated text
plt.figure(figsize=(10, 6))
plt.title("Recursive Text Expansion")
plt.plot(generated_text, marker='o', linestyle='-')
plt.xlabel("Iterations")
plt.ylabel("Generated Text")
plt.grid(True)
plt.show()

            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    