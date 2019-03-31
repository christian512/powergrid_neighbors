# Powergrid Neighborhood
This is a test project for genetic algorithms, that should optimize costs for power within a residential neighborhood. Please note that this is just a test project and mainly written for a course at Københavns Universitet. 
However it can illustrate the capabilities of genetic algorithms and also connect it to realistic problem. For more information please read the text below. The report (for the course at KU) will be uploaded later.

## Model
Implementation at [src/grid.py](https://github.com/christian512/powergrid_neighbors/blob/master/src/grid.py).
Example usage at [single_household_simulation.ipynb](https://github.com/christian512/powergrid_neighbors/blob/master/single_household_simulation.ipynb)

This is an example model, for simulating the power distribution within a small neighborhood. It consists of a fixed number of households, that have different energy consumption schedules every day. Each house can have a photovoltaic system installed on the roof and a storage grid can be introduced. This grid allows several houses to be connected to the same storage and thus share energy with neighbors. The model includes simple loss functions, that might not be realistic enough and need more sophisticated approaches. You can also set the cost for several parameters (see in implementation or examples). After setting up the power grid within the neighborhood (including pv systems and storages), one can simulate the distribution for a given timeframe and given consumption data (and production data for pv). The output includes costs such as setting up the storages and pv panels and costs/rewards for buying/selling energy to the external powergrid.

## Data 
The dataset we used in this project to simulate our chosen configuration of the neighborhood grid configuration is provided by Ausgrid. [Ausgrid](https://www.ausgrid.com.au/) is an Australian electricity distribution company that shared the dataset for public use on their [website](https://www.ausgrid.com.au/Industry/Innovation-and-research/Data-to-share/Solar-home-electricity-data).
However the implementation, described before, is able to handle other input data. For this project, I wanted to keep the dataset close to reality. The Ausgrid data set includes consumption/production measurements for 300 houses for one year (at a half-hourly frequency).
Some basic properties of the data set are show in [data_visualisation.ipynb](https://github.com/christian512/powergrid_neighbors/blob/master/data_visualisation.ipynb).

## Optimization of the storage system
Adding a storage system, that allows households to share their produced power, introduces interaction between the houses. In order to optimize the objective function ( costs in a specific timeframe) we use a genetic algorithm, to change connections between houses and storages and the capacities of the storages. This algorithm optimizes also with regards to the losses in the system (longer connections have higher losses than shorter ones). The current implementation of the algorithm can be found at [strategy2_storage_optimization.ipynb](https://github.com/christian512/powergrid_neighbors/blob/master/strategy2_storage_optimization.ipynb).


## Optimization of the pv system sizes 
Using this model one can look at any individual house (and its consumption schedule) and check its current pv system installed. Since Ausgrid also provides the sizes of the pv panels, one can check if a bigger/smaller installation would be more efficient to use. Since this is just a simple model I assume linear scaling of the production with the size. I also do not consider any space limitations on the roof of the house. A simple introduction to that optimization is given at [pv_size_optimization.ipynb](https://github.com/christian512/powergrid_neighbors/blob/master/pv_size_optimization.ipynb)

