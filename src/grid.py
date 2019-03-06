import numpy as np
import random

class Grid:
    """
    Class that represents the neighborhood grid and
    includes various operations such as changing grid lines.
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

    def print(self):
        """Prints certain information"""
        print('pv_types: ' + str(self._house_pv_type))
        print('storag_conns: ' + str(self._house_storage_connections))


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
