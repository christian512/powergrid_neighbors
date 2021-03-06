import numpy as np
import pandas as pd
import random
import sys

class Grid:
    """
    This class implements the a configuration of a neighborhood with all properties. Namely the are
    pv systems, storages and connections of houses to the storages. The class also implements all
    needed functionality to edit the configuration of the neighborhood in order to implement a genetic algorithm.
    """

    def __init__(self,num_houses=10, num_storages=1, max_capacity=5, num_pvtypes=1, pv_peakpower=1):
        """
        Initializes the neighborhood grid.
        :param num_houses: Number of houses included in the neighborhood
        :param num_storages: Number of storages in the neighborhood
        :param max_capacity: Gives the maximum capacity [kWh] for the storages / either list for individual or
                             float for everyone
                             This also gives the upper boundary for mutations of storage sizes
        :param num_pvtypes: Number of different pvtypes available
        :param pv_peakpower: The peak power for all pv types [kWp], if only integer it will be changed to list (as max_capacity)
        """
        # Check the dimensions:
        assert num_houses > 0
        assert num_storages > 0
        assert num_pvtypes > -1
        if isinstance(max_capacity,list) or isinstance(max_capacity,np.ndarray):
            assert len(max_capacity) == num_storages
        else:
            max_capacity = np.array([max_capacity]*num_storages,dtype=float)
        if isinstance(pv_peakpower,list) or isinstance(pv_peakpower,np.ndarray):
            assert len(pv_peakpower) == num_pvtypes
        else:
            pv_peakpower = np.array([pv_peakpower]*num_pvtypes,dtype=float)
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
        # Position of each storage
        self._storage_pos = np.linspace(0,self._num_houses-1,self._num_storages,dtype=int)
        # Performance of single pv types
        self._peak_power_pv = pv_peakpower

        # Set indicator for cost settings and list all prices
        self._set_costs = False
        self._cost_kwh_grid_import = 0 # Price per kWh imported from power grid
        self._gain_kwh_grid_export = 0 # Price per kWh exported to the power grid
        self._cost_storage_per_kwh = 0 # Price per kWh in storage system
        self._cost_pv_per_kwp = 0
        self._cost_per_wire = 0
        self._loss_per_unit = 0

    def set_costs(self,kwh_import=0.25,kwh_export=0.10,cost_storage_kwh=500,cost_pv_kwp=1400,cost_per_wire=500,loss_per_unit=0.1):
        """
        Setting the costs, for different quantities listed below (all prices in EURO)
        :param kwh_import: Price per kwh imported
        :param kwh_export: Amount you get for exporting one kWH
        :param cost_storage_kwh: Price per kwh in storage
        :param cost_pv_kwp: Price per kWp for solar panels
        :param cost_per_wire: Cost per wire connection from one house to its direct neighbor
        :param loss_per_unit: Loss of energy from transferring energy from one house to another
        """
        self._cost_kwh_grid_import = kwh_import
        self._gain_kwh_grid_export = kwh_export
        self._cost_storage_per_kwh = cost_storage_kwh
        self._cost_pv_per_kwp = cost_pv_kwp
        self._cost_per_wire = cost_per_wire
        self._loss_per_unit = loss_per_unit

        # indicate that prices are set
        self._set_costs = True

    def randomize(self,storage_connections=True,pv_types=True):
        """
        Randomizes the setup of the current neighborhood
        :param storage_connections : Should storage connections be randomized
        :param pv_types : Should pv types be randomized
        """
        for i in range(self._num_houses):
            if storage_connections:
                self._house_storage_connections[i] = int((self._num_storages-1) * random.random())
            if pv_types:
                self._house_pv_type[i] = int((self._num_pvtypes + 1) * random.random()) - 1

    def get_copy(self):
        """Returns a copy of itself."""
        g_copy = Grid(num_houses=self._num_houses,
                      num_storages=self._num_storages,
                      max_capacity=np.copy(self._max_capacities_storages),
                      num_pvtypes=self._num_pvtypes,
                      pv_peakpower=np.copy(self._peak_power_pv)
                    )
        g_copy._house_storage_connections = np.copy(self._house_storage_connections)
        g_copy._storage_pos = np.copy(self._storage_pos)
        g_copy._house_pv_type = np.copy(self._house_pv_type)
        g_copy._charge_level_storages = np.copy(self._charge_level_storages)
        g_copy._set_costs = self._set_costs
        g_copy._cost_kwh_grid_import = self._cost_kwh_grid_import
        g_copy._gain_kwh_grid_export = self._gain_kwh_grid_export
        g_copy._cost_storage_per_kwh = self._cost_storage_per_kwh
        g_copy._cost_pv_per_kwp = self._cost_pv_per_kwp
        g_copy._cost_per_wire = self._cost_per_wire
        g_copy._loss_per_unit = self._loss_per_unit
        return g_copy

    def mutate(self,num_house=-1,num_storage=-1,storage_connection=False,storage_sizes=False,pv_type=False,storage_cap_list=[]):
        """
        Mutates the storage connection and type of installed pv of ONE house
        :param num_house: Number of house to mutate, if -1 it is random
        :param num_storage: Number of storage to be mutated, only for storage_sizes=True
        :param storage_connection: Bool if storage connection should be mutated
        :param storage_sizes: Bool if storage sizes should be mutated
        :param pv_type: Bool if pv_type should be mutated
        :param storage_cap_list: List of storage capacities to choose from if storage_sizes == True
        """
        # Assert that we have more then one possibility

        # Generate random house number if -1
        if num_house == -1:
            num_house = int(self._num_houses * random.random())
        # Mutate storage connection
        if storage_connection and self._num_storages > 1:
            assert self._num_storages >= 1, 'Mutation of storage not possible: Only one storage available'
            old_storage_connection = self._house_storage_connections[num_house]
            while old_storage_connection == self._house_storage_connections[num_house]:
                self._house_storage_connections[num_house] = int(self._num_storages * random.random())

        if storage_sizes:
            if num_storage == -1 or num_storage not in self._house_storage_connections:
                num_storage = int(self._num_storages * random.random())
                while num_storage not in self._house_storage_connections:
                    num_storage = int(self._num_storages * random.random())
            # Set new storage size which should be max capacity at maximum
            if len(storage_cap_list) > 0:
                self._max_capacities_storages[num_storage] = random.choice(list(storage_cap_list))
            if len(storage_cap_list) == 0:
                print('Need cap list')
                #self._max_capacities_storages[num_storage] = int(self._max_capacities_storages[num_storage] * random.random())


        # Mutate pv type
        if pv_type:
            assert self._num_pvtypes > 0, 'Mutation of pvtype not possible: No pv_types available'
            old_pv_type = self._house_pv_type[num_house]
            while old_pv_type == self._house_pv_type[num_house]:
                self._house_pv_type[num_house] = int((self._num_pvtypes + 1) * random.random()) - 1

    def crossover(self,other_grid,storage_connection=False,storage_sizes=False,pv_type=False,pos=[-1,-1,-1]):
        """
        Crosses the properties of one grid with another
        :param other_grid: The other grid to crossover with
        :param storage_connection: Boolean should storage connections be crossed
        :param pv_type: Boolean should pv_types be crossed
        :param storage_sizes: Bool should storage_sizes be crossed
        :param pos: Position of crossover one for storage connection, sizes and pv_type / -1 for random
        :return: other_grid crossover with this grid
        """
        assert self._num_houses == other_grid._num_houses
        assert self._num_pvtypes == other_grid._num_pvtypes

        if storage_connection:
            # Get point for crossover
            if pos[0] == -1:
                pos[0] = int((self._num_houses-1)*random.random())
            assert pos[0] < self._num_houses
            # Crossover
            tmp = np.copy(self._house_storage_connections[pos[0]:])
            self._house_storage_connections[pos[0]:] = np.copy(other_grid._house_storage_connections[pos[0]:])
            other_grid._house_storage_connections[pos[0]:] = np.copy(tmp)

        if storage_sizes:
            # Get point for crossover
            if pos[1] == -1:
                pos[1] = int((self._num_storages-1)*random.random())
            assert pos[1] < self._num_storages
            # Crossover
            tmp = np.copy(self._max_capacities_storages[pos[1]:])
            self._max_capacities_storages[pos[1]:] = np.copy(other_grid._max_capacities_storages[pos[1]:])
            other_grid._max_capacities_storages[pos[1]:] = np.copy(tmp)


        if pv_type and self._num_pvtypes > 0:
            # Get point for crossover
            if pos[2] == -1:
                pos[2] = int((self._num_pvtypes-1)*random.random())
            assert pos[2] < self._num_pvtypes
            # Crossover
            tmp = np.copy(self._house_pv_type[pos[2]:])
            self._house_pv_type[pos[2]:] = np.copy(other_grid._house_pv_type[pos[2]:])
            other_grid._house_pv_type[pos[2]:] = np.copy(tmp)

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

    def change_storages(self, num_storages=1, max_capacity = 0):
        """
        Changes the storage network of the grid model
        :param num_storages: Number of storages in the grid
        :param max_capacity: maximum capacity for each storage either list or int
        """
        assert num_storages > 0
        # List of all max capacites
        if isinstance(max_capacity,list) or isinstance(max_capacity,np.ndarray):
            assert len(max_capacity) == num_storages
        else:
            max_capacity = np.array([max_capacity]*num_storages)

        # Set class vars
        self._num_storages = num_storages
        self._charge_level_storages = np.zeros(self._num_storages, dtype=float)
        self._max_capacities_storages = max_capacity
        self._storage_pos = np.linspace(0,self._num_houses,self._num_storages,dtype=int,endpoint=False)

        # Random connections to storages
        for i in range(self._num_houses):
            self._house_storage_connections[i] = int((self._num_storages-1) * random.random())

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
        print('pv_peakpowers: ' + str(self._peak_power_pv))
        print('storag_conns: ' + str(self._house_storage_connections))
        print('storage_capacities: ' + str(self._max_capacities_storages))
        print('storage_positions:' + str(self._storage_pos))
        
    def simulate(self, data_cons, data_prod):
        """
        Function that simulates the neighborhood energyflow for a given timeinterval.
        It iterates through time and checks for each house the demand/production.
        If a house demands energy and it is in the storage it just takes this, otherwise it buys it from the grid.
        If a house has a overproduction than it stores the energy in the storage if it is empty. Otherwise
        it sells the power to the grid.
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
        if self._num_pvtypes > 0: assert data_cons.shape[0] == data_prod.shape[0], \
            'grid.simulate(): DataFrames of prod and cons need same length!'

        # Create a dictionary for output
        res_dict = {
            "import_grid_kwh" : 0.0,
            "export_grid_kwh" : 0.0,
            "pv_production_kwh": 0.0,
            "setup_cost_storage" : 0.0,
            "setup_cost_pv" : 0.0,
            "cost_import_grid" : 0.0,
            "reward_export_grid" : 0.0
        }

        # Reshaping in the case of only a single house or single pv type
        if self._num_houses == 1: data_cons = data_cons.reshape(data_cons.shape[0],1)
        if self._num_pvtypes == 1: data_prod = data_prod.reshape(data_prod.shape[0],1)

        # Calculate initial costs
        if not self._set_costs: sys.exit('Set costs before simulating the grid!')
        # Check which storages are used
        used_storages = np.unique(self._house_storage_connections)
        all_storages = np.arange(self._num_storages)
        mask = np.isin(all_storages,used_storages,invert=True)
        not_used_storages = all_storages[mask]
        org_max_capacities = np.copy(self._max_capacities_storages)
        for k in not_used_storages: self._max_capacities_storages[k] = 0
        res_dict["setup_cost_storage"] = np.sum(self._max_capacities_storages)*self._cost_storage_per_kwh
        # Sum all kWp ours installed on the houses
        for k in self._house_pv_type:
            if k == -1:
                continue
            res_dict["setup_cost_pv"] += self._peak_power_pv[k]*self._cost_pv_per_kwp

        # Loop over all time steps
        for i in range(data_cons.shape[0]):
            # Take each house
            for house_num in range(self._num_houses):
                # Get PV type, connected storage number, storage position
                pv_type = self._house_pv_type[house_num]
                storage_num = self._house_storage_connections[house_num]
                storage_pos = self._storage_pos[storage_num]
                loss = self.__get_loss(storage_pos,house_num)
                # Get energy demand of house
                demand = data_cons[i, house_num]
                # Check if there is energy coming from PV
                if pv_type > -1:
                    # Get the energy production of that PV
                    res_dict["pv_production_kwh"] += data_prod[i,pv_type]
                    demand = demand - data_prod[i, pv_type]

                # If the house needs energy and enough energy is in storage
                if self._charge_level_storages[storage_num] >= demand*(1+loss) and demand > 0:
                    self._charge_level_storages[storage_num] -= demand*(1+loss)
                # If house need energy and storage only has partial energy
                elif self._charge_level_storages[storage_num] < demand*(1+loss) and demand > 0:
                    demand -= self._charge_level_storages[storage_num]*(1-loss)
                    self._charge_level_storages[storage_num] = 0
                    res_dict["import_grid_kwh"] += demand
                # If house has a over production
                elif demand < 0:
                    production = -1*demand
                    # If storage has enough space for the whole energy
                    if production*(1-loss) <= \
                            self._max_capacities_storages[storage_num] - self._charge_level_storages[storage_num]:
                        self._charge_level_storages[storage_num] += production*(1-loss)
                    # If there is not enough space in the energy storage system
                    else:
                        production -= (self._max_capacities_storages[storage_num] -
                                       self._charge_level_storages[storage_num])*(1+loss)
                        self._charge_level_storages[storage_num] = self._max_capacities_storages[storage_num]
                        res_dict["export_grid_kwh"] += production

        # Reset charge levels to zero
        self._charge_level_storages = np.zeros(self._num_storages, dtype=float)
        # Set back the to inital max capacities
        self._max_capacities_storages = np.copy(org_max_capacities)
        # Calculate the expenses and reward for importing and exporting
        res_dict["cost_import_grid"] = res_dict["import_grid_kwh"]*self._cost_kwh_grid_import
        res_dict["reward_export_grid"] = res_dict["export_grid_kwh"] * self._gain_kwh_grid_export
        # Return results
        return res_dict

    def __get_loss(self,storage_pos,house_pos):
        """
        Calculates the loss between a house and a storage at given positions
        :param storage_pos: Position of the storage
        :param house_pos: Position of the house
        :return: loss
        """
        dist = 0
        if storage_pos == house_pos: dist =  0
        pos1 = max(storage_pos,house_pos)
        pos2 = min(storage_pos,house_pos)
        if pos1 % 2 == pos2 % 2: dist = (pos1-pos2)/2
        elif pos1 % 2 == 1: dist = (pos1-pos2+1)/2
        elif pos1 % 2 == 0: dist = (pos1-pos2+1)/2

        return np.min([dist*self._loss_per_unit,1])
        
    def get_total_distance(self):
        """Calculates the total number of wires need in the grid"""
        # Create a list for each storages position
        pos_store = np.zeros(self._num_houses)
        for i in range(self._house_storage_connections.shape[0]):
            pos_store[i] = self._storage_pos[self._house_storage_connections[i]]
        return np.sum(np.abs(np.arange(self._num_houses)-pos_store))
    


if __name__ == '__main__':
    # Some basic tests
    g1 = Grid(num_houses=10,num_storages=1,max_capacity=3,num_pvtypes=1)
    g2 = g1.get_copy()
    g1.change_storages(num_storages=3,max_capacity=3)
    g1.print()
    g2.print()
