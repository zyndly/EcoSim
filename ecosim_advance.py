import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import random

# Define grid states
EMPTY = 0
PREY = 1
PREDATOR = 2
VEGETATION = 3
WATER = 4
SUPER_PREDATOR = 5

class EcosystemSimulation:
    def __init__(self, grid_size=30):
        self.grid_size = grid_size
        self.grid = self.initialize_grid()
        self.environmental_events = {
            'drought_probability': 0.05,
            'flood_probability': 0.03,
            'disease_probability': 0.02
        }

    def initialize_grid(self):
        # More nuanced initial grid generation
        grid = np.random.choice(
            [EMPTY, PREY, PREDATOR, VEGETATION, WATER, SUPER_PREDATOR], 
            size=(self.grid_size, self.grid_size), 
            p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
        )
        return grid

    def get_neighbors(self, x, y):
        # Get valid neighboring cells
        neighbors = []
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def apply_environmental_event(self, drought, flood, disease):
        # Random environmental events
        if drought and random.random() < self.environmental_events['drought_probability']:
            # Drought reduces vegetation and impacts prey
            drought_mask = np.random.random(self.grid.shape) < 0.3
            self.grid[drought_mask & (self.grid == VEGETATION)] = EMPTY
            self.grid[drought_mask & (self.grid == PREY)] = EMPTY
        
        if flood and random.random() < self.environmental_events['flood_probability']:
            # Flood reduces populations in lower grid areas
            flood_mask = np.random.random(self.grid.shape) < 0.2
            self.grid[flood_mask] = WATER
        
        if disease and random.random() < self.environmental_events['disease_probability']:
            # Disease outbreak randomly reduces population
            disease_mask = np.random.random(self.grid.shape) < 0.15
            self.grid[disease_mask & ((self.grid == PREY) | (self.grid == PREDATOR))] = EMPTY

    def simulate_step(self, drought, flood, disease):
        # Create a copy of the grid to modify
        new_grid = self.grid.copy()

        # Apply environmental events
        self.apply_environmental_event(drought, flood, disease)

        # Simulate for each cell
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.grid[x, y]
                
                if cell == PREY:
                    # Prey reproduction and movement
                    neighbors = self.get_neighbors(x, y)
                    empty_neighbors = [n for n in neighbors if self.grid[n[0], n[1]] == EMPTY]
                    
                    # Reproduction
                    if empty_neighbors and random.random() < 0.01:  # Further reduced reproduction chance
                        nx, ny = random.choice(empty_neighbors)
                        new_grid[nx, ny] = PREY
                    
                    # High mortality rate
                    if random.random() < 0.5:  # Increased mortality rate to 50%
                        new_grid[x, y] = EMPTY
                        # Two predators and two super predators die after prey dies
                        predator_deaths = 0
                        super_predator_deaths = 0
                        for n in neighbors:
                            if self.grid[n[0], n[1]] == PREDATOR and predator_deaths < 2:
                                new_grid[n[0], n[1]] = EMPTY
                                predator_deaths += 1
                            elif self.grid[n[0], n[1]] == SUPER_PREDATOR and super_predator_deaths < 2:
                                new_grid[n[0], n[1]] = EMPTY
                                super_predator_deaths += 1
                
                elif cell == PREDATOR:
                    # Predator hunting and reproduction
                    neighbors = self.get_neighbors(x, y)
                    prey_neighbors = [n for n in neighbors if self.grid[n[0], n[1]] == PREY]
                    
                    if prey_neighbors:
                        # Hunt prey
                        nx, ny = random.choice(prey_neighbors)
                        new_grid[nx, ny] = PREDATOR
                        new_grid[x, y] = EMPTY
                        
                        # Reproduction
                        if random.random() < 0.25:  # Increased reproduction chance
                            empty_neighbors = [n for n in neighbors if self.grid[n[0], n[1]] == EMPTY]
                            if empty_neighbors:
                                nx, ny = random.choice(empty_neighbors)
                                new_grid[nx, ny] = PREDATOR
                    elif random.random() < 0.15:  # Reduced starvation rate
                        # Starvation
                        new_grid[x, y] = EMPTY
                    
                    # Predator can be eaten by super predators
                    super_predator_neighbors = [n for n in neighbors if self.grid[n[0], n[1]] == SUPER_PREDATOR]
                    if super_predator_neighbors:
                        new_grid[x, y] = EMPTY
                
                elif cell == SUPER_PREDATOR:
                    # Super predator hunting and reproduction
                    neighbors = self.get_neighbors(x, y)
                    huntable_neighbors = [n for n in neighbors if self.grid[n[0], n[1]] in [PREY, PREDATOR]]
                    
                    if huntable_neighbors:
                        nx, ny = random.choice(huntable_neighbors)
                        new_grid[nx, ny] = SUPER_PREDATOR
                        new_grid[x, y] = EMPTY
                        
                        # Reproduction
                        if random.random() < 0.5:  # Increased reproduction chance to twice that of predators
                            empty_neighbors = [n for n in neighbors if self.grid[n[0], n[1]] == EMPTY]
                            if empty_neighbors:
                                nx, ny = random.choice(empty_neighbors)
                                new_grid[nx, ny] = SUPER_PREDATOR
                    elif random.random() < 0.15:  # Reduced starvation rate
                        # Starvation
                        new_grid[x, y] = EMPTY

        # Update grid
        self.grid = new_grid
        return self.grid

def main():
    st.title('Advanced EcoSim: Ecosystem Dynamics Simulator')

    # Sidebar for simulation parameters
    st.sidebar.header('Simulation Parameters')
    grid_size = st.sidebar.slider('Grid Size', 20, 50, 30)
    steps = st.sidebar.slider('Simulation Steps', 1, 100, 20)

    # Environmental event toggles
    drought = st.sidebar.checkbox('Enable Drought Events', value=True)
    flood = st.sidebar.checkbox('Enable Flood Events', value=True)
    disease = st.sidebar.checkbox('Enable Disease Events', value=True)

    # Initialize simulation
    sim = EcosystemSimulation(grid_size)

    # Simulation steps and data collection
    population_data = []
    grid_states = []

    # Create a Streamlit placeholder for the grid visualization
    grid_placeholder = st.empty()

    # Run simulation
    for step in range(steps):
        # Simulate and update the grid
        grid = sim.simulate_step(drought, flood, disease)
        grid_states.append(grid.copy())

        # Visualize the current grid state
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid, cmap='viridis')
        plt.colorbar(im, ax=ax, ticks=[EMPTY, PREY, PREDATOR, VEGETATION, WATER, SUPER_PREDATOR], 
                     label='Habitat State')
        plt.title(f'Ecosystem Grid - Step {step}')
        
        # Update the placeholder with the new visualization
        grid_placeholder.pyplot(fig)
        plt.close(fig)

        # Track population data
        prey_count = np.count_nonzero(grid == PREY)
        predator_count = np.count_nonzero(grid == PREDATOR)
        super_predator_count = np.count_nonzero(grid == SUPER_PREDATOR)
        vegetation_count = np.count_nonzero(grid == VEGETATION)
        water_count = np.count_nonzero(grid == WATER)
        population_data.append({
            'step': step, 
            'prey': prey_count, 
            'predator': predator_count, 
            'super_predator': super_predator_count,
            'vegetation': vegetation_count,
            'water': water_count
        })

    # Convert to DataFrame
    population_df = pd.DataFrame(population_data)

    # Visualization Section
    st.header('Ecosystem Simulation Results')

    # Population Dynamics Plot
    st.subheader('Population Dynamics')
    fig, ax = plt.subplots(figsize=(12, 6))
    color_map = {
        'prey': 'green',
        'predator': 'red', 
        'super_predator': 'purple',
        'vegetation': 'lime',
        'water': 'blue'
    }
    
    for species, color in color_map.items():
        ax.plot(population_df['step'], population_df[species], 
                label=species.replace('_', ' ').title(), 
                color=color, 
                marker='o')
    
    ax.set_xlabel('Simulation Steps')
    ax.set_ylabel('Population Count')
    ax.set_title('Ecosystem Population Dynamics')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    plt.close(fig)

    # Advanced Statistics
    st.subheader('Ecosystem Health Metrics')
    
    # Biodiversity and Stability Metrics
    total_population = population_df[['prey', 'predator', 'super_predator', 'vegetation']].sum()
    biodiversity_index = total_population.mean()
    population_stability = population_df[['prey', 'predator', 'super_predator']].std().mean()

    # Visualization of Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Biodiversity Index
    ax1.bar(['Biodiversity Index'], [biodiversity_index], color='teal')
    ax1.set_title('Ecosystem Biodiversity')
    ax1.set_ylabel('Index Value')
    
    # Population Stability
    ax2.bar(['Population Stability'], [population_stability], color='coral')
    ax2.set_title('Population Variability')
    ax2.set_ylabel('Standard Deviation')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Detailed Statistics Table
    st.subheader('Detailed Ecosystem Statistics')
    stats_df = pd.DataFrame({
        'Metric': [
            'Total Population', 
            'Biodiversity Index', 
            'Population Stability',
            'Prey Population', 
            'Predator Population',
            'Super Predator Population',
            'Vegetation Cover'
        ],
        'Value': [
            total_population.sum(),
            biodiversity_index,
            population_stability,
            population_df['prey'].mean(),
            population_df['predator'].mean(),
            population_df['super_predator'].mean(),
            population_df['vegetation'].mean()
        ]
    })
    st.table(stats_df)

if __name__ == "__main__":
    main()