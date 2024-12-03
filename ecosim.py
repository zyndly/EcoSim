import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Define grid size
grid_size = 20

# Initialize the grid with random prey (1) and predators (2)
grid = np.random.choice([0, 1, 2, 3, 4, 5], size=(grid_size, grid_size), p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05])

def simulate_step(grid):
    new_grid = grid.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 1:  # Prey
                # Prey reproduction
                if np.random.rand() < 0.05:
                    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                    for ni, nj in neighbors:
                        if 0 <= ni < grid_size and 0 <= nj < grid_size and new_grid[ni, nj] == 0:
                            new_grid[ni, nj] = 1
                            break
            elif grid[i, j] == 2:  # Predator
                # Predator hunting
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in neighbors:
                    if 0 <= ni < grid_size and 0 <= nj < grid_size and new_grid[ni, nj] == 1:
                        new_grid[ni, nj] = 2
                        new_grid[i, j] = 0
                        break
                else:
                    # Predator starvation
                    if np.random.rand() < 0.1:
                        new_grid[i, j] = 0
            elif grid[i, j] == 5:  # New predator type
                # New predator hunting
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in neighbors:
                    if 0 <= ni < grid_size and 0 <= nj < grid_size and new_grid[ni, nj] in [1, 2]:
                        new_grid[ni, nj] = 5
                        new_grid[i, j] = 0
                        break
                else:
                    # New predator starvation
                    if np.random.rand() < 0.1:
                        new_grid[i, j] = 0
    return new_grid

# Streamlit app
st.title('EcoSim: A Simulated Ecosystem')

# User input for grid size
grid_size = st.slider('Grid Size', 10, 50, 20)

# Initialize the grid
grid = np.random.choice([0, 1, 2, 3, 4, 5], size=(grid_size, grid_size), p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05])

# Display the initial grid
st.write('Initial Ecosystem Grid')
st.write(grid)

# Simulate steps
steps = st.slider('Number of Steps', 1, 100, 10)
population_data = []

for step in range(steps):
    grid = simulate_step(grid)
    prey_count = np.count_nonzero(grid == 1)
    predator_count = np.count_nonzero(grid == 2)
    new_predator_count = np.count_nonzero(grid == 5)
    population_data.append({'step': step, 'prey': prey_count, 'predator': predator_count, 'new_predator': new_predator_count})

# Convert to DataFrame
population_df = pd.DataFrame(population_data)

# Display population trends
st.write('Population Trends Over Time')
st.line_chart(population_df.set_index('step'))

# Display the final grid
st.write('Final Ecosystem Grid')
st.write(grid)

# Calculate advanced statistics
def calculate_statistics(population_df):
    # Biodiversity index
    population_df['biodiversity_index'] = population_df['prey'] + population_df['predator'] + population_df['new_predator']
    
    # Species survival probability
    population_df['survival_probability'] = population_df['prey'] / (population_df['prey'] + population_df['predator'] + population_df['new_predator'])
    
    return population_df

# Calculate and display advanced statistics
population_df = calculate_statistics(population_df)
st.write('Advanced Statistics')
st.write(population_df[['biodiversity_index', 'survival_probability']])

# Enhanced animation using Streamlit
def animate_grid(grid):
    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap='viridis', animated=True)
    plt.colorbar(im, ticks=[0, 1, 2, 3, 4, 5], label='Habitat State')
    plt.title('Ecosystem Grid')
    
    def update(frame):
        grid = simulate_step(grid)
        im.set_array(grid)
        return im,
    
    ani = FuncAnimation(fig, update, frames=range(steps), blit=True)
    return ani

# Display animation
st.write('Ecosystem Animation')
ani = animate_grid(grid)
st.pyplot(ani.to_jshtml())