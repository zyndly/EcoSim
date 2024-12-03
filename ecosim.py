import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def simulate_step(grid):
    grid_size = grid.shape[0]
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

def main():
    st.title('EcoSim: A Simulated Ecosystem')

    # User input for grid size (modified min and max)
    grid_size = st.slider('Grid Size', 20, 50, 30)

    # Initialize the grid
    grid = np.random.choice([0, 1, 2, 3, 4, 5], size=(grid_size, grid_size), p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05])

    # Display the initial grid
    st.subheader('Initial Ecosystem Grid')
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, cmap='viridis')
    plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5], 
                 label='Habitat State')
    plt.title('Initial Ecosystem Grid')
    st.pyplot(fig)
    plt.close(fig)

    # Simulate steps
    steps = st.slider('Number of Steps', 1, 100, 10)
    population_data = []

    # Create a Streamlit placeholder for the grid visualization
    grid_placeholder = st.empty()

    for step in range(steps):
        # Simulate and update the grid
        grid = simulate_step(grid)
        
        # Visualize the current grid state
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid, cmap='viridis')
        plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5], 
                     label='Habitat State')
        plt.title(f'Ecosystem Grid - Step {step}')
        
        # Update the placeholder with the new visualization
        grid_placeholder.pyplot(fig)
        plt.close(fig)

        # Track population data
        prey_count = np.count_nonzero(grid == 1)
        predator_count = np.count_nonzero(grid == 2)
        new_predator_count = np.count_nonzero(grid == 5)
        population_data.append({
            'step': step, 
            'prey': prey_count, 
            'predator': predator_count, 
            'new_predator': new_predator_count
        })

    # Convert to DataFrame
    population_df = pd.DataFrame(population_data)

    # Display population trends with enhanced visualization
    st.subheader('Population Trends Over Time')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(population_df['step'], population_df['prey'], label='Prey', color='green', marker='o')
    ax.plot(population_df['step'], population_df['predator'], label='Predator', color='red', marker='s')
    ax.plot(population_df['step'], population_df['new_predator'], label='New Predator', color='purple', marker='^')
    ax.set_xlabel('Simulation Steps')
    ax.set_ylabel('Population Count')
    ax.set_title('Population Dynamics in EcoSim')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    plt.close(fig)

    # Display the final grid with enhanced visualization
    st.subheader('Final Ecosystem Grid')
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(grid, cmap='viridis')
    plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5], 
                 label='Habitat State')
    plt.title('Final Ecosystem Grid')
    st.pyplot(fig)
    plt.close(fig)

    # Advanced statistics with enhanced visualization
    def calculate_statistics(population_df):
        population_df['biodiversity_index'] = (
            population_df['prey'] + 
            population_df['predator'] + 
            population_df['new_predator']
        )
        population_df['survival_probability'] = population_df['prey'] / (
            population_df['prey'] + 
            population_df['predator'] + 
            population_df['new_predator']
        )
        return population_df

    # Calculate and display advanced statistics
    population_df = calculate_statistics(population_df)
    
    st.subheader('Advanced Statistics')
    
    # Create a more appealing visualization of advanced statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Biodiversity Index
    ax1.bar(['Biodiversity Index'], [population_df['biodiversity_index'].mean()], color='teal')
    ax1.set_title('Average Biodiversity Index')
    ax1.set_ylabel('Index Value')
    
    # Survival Probability
    ax2.bar(['Survival Probability'], [population_df['survival_probability'].mean()], color='coral')
    ax2.set_title('Average Survival Probability')
    ax2.set_ylabel('Probability')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Detailed statistics table
    st.write('Detailed Statistics:')
    stats_df = pd.DataFrame({
        'Metric': ['Total Population', 'Average Biodiversity', 'Average Survival Probability'],
        'Value': [
            population_df[['prey', 'predator', 'new_predator']].sum().sum(),
            population_df['biodiversity_index'].mean(),
            population_df['survival_probability'].mean()
        ]
    })
    st.table(stats_df)

if __name__ == "__main__":
    main()