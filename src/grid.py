import numpy as np
import pandas as pd
import random

class Grid:
    """
    Class that represents the neighborhood grid and
    includes various operations such as changing grid lines.
    # TODO: function that calculates the different cost quantities for a specific timeinterval
    """

    def __init__(self,num_houses=10, num_storages=1,max_capacity=5,num_pvtypes=1):
        """
        Initializes the neighborhood grid.
        :param num_houses: Number of houses included in the neighborhood
        :param num_storages: Number of storages in the neighborhood
        :param max_capacity: Gives the maximum capacity for the storages / either list for individual or float for everyone
        :param num_pvtypes: Number of different pvtypes available
        """
        # Check the dimensions:
        assert num_houses > 0
        assert num_storages > 0
        assert num_pvtypes > -1
        if isinstance(max_capacity,list) or isinstance(max_capacity,np.ndarray):
            assert len(max_capacity) == num_storages
        else:
            max_capacity = np.array([max_capacity]*num_storages)

        # Store variables in object
        self._num_houses = num_houses
        self._num_storages = num_storages
        self._num_pvtypes = num_pvtypes

        # Connections of each house to storage
        self._house_storage_connections = np.zeros(self._num_houses, dtype=int)
        # Type of PV installed on the house, -1 means None
        self._house_pv_type = np.zeros(self._num_houses, dtype=int) - 1
        # Charging level of each storage in kWh
        self._charge_level_storages = np.zeros(self._num_storages, dtype=float)
        # Maximum capacity of each storage in kWh
        self._max_capacities_storages = max_capacity

    def randomize(self):
        """
        Randomizes the setup of the current neighborhood
        """
        for i in range(self._num_houses):
            self._house_storage_connections[i] = int(self._num_storages * random.random())
            self._house_pv_type[i] = int((self._num_pvtypes + 1) * random.random()) - 1

    def mutate(self,num_house=-1,storage_connection=True,pv_type=True):
        """
        Mutates the storage connection and type of installed pv
        :param num_house: Number of house to mutate, if -1 it is random
        :param storage_connection: Bool if storage connection should be mutated
        :param pv_type: Bool if pv_type should be mutated
        """
        # Assert that we have more then one possibility

        # Generate random house number if -1
        if num_house == -1:
            num_house = int(self._num_houses * random.random())
        # Mutate storage connection
        if storage_connection:
            assert self._num_storages > 1, 'Mutation of storage not possible: Only one storage available'
            old_storage_connection = self._house_storage_connections[num_house]
            while old_storage_connection == self._house_storage_connections[num_house]:
                self._house_storage_connections[num_house] = int(self._num_storages * random.random())
        # Mutate pv type
        if pv_type:
            assert self._num_pvtypes > 0, 'Mutation of pvtype not possible: No pv_types available'
            old_pv_type = self._house_pv_type[num_house]
            while old_pv_type == self._house_pv_type[num_house]:
                self._house_pv_type[num_house] = int((self._num_pvtypes + 1) * random.random()) - 1

    def crossover(self,other_grid,storage_connection=True,pv_type=True,pos=[-1,-1]):
        """
        Crosses the properties of one grid with another
        :param other_grid: The other grid to crossover with
        :param storage_connection: Boolean should storage connections be crossed
        :param pv_type: Boolean should pv_types be crossed
        :param pos: Position of crossover one for storage one for pv_type / -1 for random
        :return: other_grid crossover with this grid
        """
        assert self._num_houses == other_grid._num_houses
        assert self._num_pvtypes == other_grid._num_pvtypes

        if storage_connection:
            # Get point for crossover
            if pos[0] == -1:
                pos[0] = int(self._num_houses*random.random())
            assert pos[0] < self._num_houses
            # Crossover
            tmp = np.copy(self._house_storage_connections[pos[0]:])
            self._house_storage_connections[pos[0]:] = np.copy(other_grid._house_storage_connections[pos[0]:])
            other_grid._house_storage_connections[pos[0]:] = np.copy(tmp)

        if pv_type and self._num_pvtypes > 0:
            # Get point for crossover
            if pos[1] == -1:
                pos[1] = int(self._num_pvtypes*random.random())
            assert pos[1] < self._num_pvtypes
            # Crossover
            tmp = np.copy(self._house_pv_type[pos[1]:])
            self._house_pv_type[pos[1]:] = np.copy(other_grid._house_pv_type[pos[1]:])
            other_grid._house_pv_type[pos[1]:] = np.copy(tmp)

        return other_grid

    def change_pvtype(self,num_house=0,pv_type=-1):
        """
        Changes the pv_type of a specific house
        :param num_house: Number of house  to change the pv type of
        :param pv_type: PV type to set house to
        """
        assert pv_type < self._num_pvtypes, 'Choose a PV type within the range of available pv types '
        assert num_house < self._num_houses, 'Choose a house within the range of available houses'
        self._house_pv_type[num_house] = pv_type

    def change_storage_connection(self,num_house=0,storage_connection=0):
        """
        Changes the storage connection of a specific house to a specific storage
        :param num_house: Number of house
        :param storage_connection: Number of storage to connect to
        """
        assert num_house < self._num_houses, 'Wrong house index'
        assert storage_connection < self._num_storages, 'Wrong storage index'
        self._house_storage_connections[num_house] = storage_connection

    def print(self):
        """Prints certain information"""
        print('pv_types: ' + str(self._house_pv_type))
        print('storag_conns: ' + str(self._house_storage_connections))

    def simulate(self, data_cons, data_prod):
        """
        Function that simulates the neighborhood energyflow for a given timeinterval.
        It iterates through time and checks for each house the demand/production.
        If a house demands energy and it is in the storage it just takes this, otherwise it buys it from the grid.
        If a house has a overproduction than it stores the energy in the storage if it is empty. Otherwise
        it sells the power to the grid.
        # TODO: Dimension check for self._num_houses == 1 is not yet perfect!!
        :param data_cons: np.ndarray that contains the consumption data for all houses
        :param data_prod: np.ndarray that contains the pv_production data for all types of pv
        :return: Dictionary with several quantities
        """
        # Check dimensions

        if self._num_houses > 1: assert data_cons.shape[1] == self._num_houses, \
            'grid.simulate(): df_cons should have consumption data for houses'
        if self._num_pvtypes > 1: assert data_prod.shape[1] == self._num_pvtypes, \
            'grid.simulate(): df_prod should hold prod data for all pv types'

        # If there are pvtypes both DataFrames should have same length
        if self._num_pvtypes > 0: assert data_cons.shape == data_prod.shape, 'DataFrames of prod and cons need same shape!'

        # Create a dictionary for output
        res_dict = {
            "import_grid" : 0.0,
            "export_grid" : 0.0,
            "pv_production": 0.0
        }

        # Reshaping in the case of only a single house or single pv type
        if self._num_houses == 1: data_cons = data_cons.reshape(data_cons.shape[0],1)
        if self._num_pvtypes == 1: data_prod = data_prod.reshape(data_prod.shape[0],1)

        # Loop over all time steps
        for i in range(data_cons.shape[0]):
            # Take each house
            for house_num in range(self._num_houses):
                # Get PV type and connected storage number
                pv_type = self._house_pv_type[house_num]
                storage_num = self._house_storage_connections[house_num]
                # Get energy demand of house
                demand = data_cons[i, house_num]
                # Check if there is energy coming from PV
                if pv_type > -1:
                    # Get the energy production of that PV
                    res_dict["pv_production"] += data_prod[i,pv_type]
                    demand = demand - data_prod[i, pv_type]

                # If the house needs energy and enough energy is in storage
                if self._charge_level_storages[storage_num] >= demand and demand > 0:
                    self._charge_level_storages[storage_num] -= demand
                # If house need energy and storage only has partial energy
                elif self._charge_level_storages[storage_num] < demand and demand > 0:
                    demand -= self._charge_level_storages[storage_num]
                    self._charge_level_storages[storage_num] = 0
                    res_dict["import_grid"] += demand
                # If house has a over production
                elif demand < 0:
                    production = -1*demand
                    # If storage has enough space for the whole energy
                    if production <= self._max_capacities_storages[storage_num] - self._charge_level_storages[storage_num]:
                        self._charge_level_storages[storage_num] += production
                    # If there is not enough space in the energy storage system
                    else:
                        production -= self._max_capacities_storages[storage_num] - self._charge_level_storages[storage_num]
                        self._charge_level_storages[storage_num] = self._max_capacities_storages[storage_num]
                        res_dict["export_grid"] += production

        return res_dict

if __name__ == '__main__':
    # Some basic tests
    g1 = Grid(num_houses=10,num_storages=1,max_capacity=3,num_pvtypes=1)
    g2 = Grid(num_houses=10, num_storages=1, max_capacity=3, num_pvtypes=1)
    g1.randomize()
    g1.print()
    g2.print()
    print('+++ Crossover +++')
    g2 = g1.crossover(g2,storage_connection=True,pv_type=True,pos=[-1,-1])
    g1.print()
    g2.print()
    print('+++ Mutation +++')
    g1.mutate(num_house=0,storage_connection=False,pv_type=True)
    g1.print()
    g2.print()
