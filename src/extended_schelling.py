"""Extended Schelling model with household formation (marriage) and openness.

This module contains the model used throughout notebooks/extended_schelling_analysis.ipynb.
The code was moved here unchanged from the first cell of the analysis notebook;
only the comments were translated into English.

Grid cell states
----------------
0  empty house
1  single agent of group A        2  single agent of group B
3  AA couple                      4  BB couple
5  AB (cross-group) couple

Main classes
------------
Person     one agent or household occupying a single cell
Schelling  the grid model (extends Cell2D) with relocation and marriage

Parameters of Schelling(n, p, m=0.1, fixed_openness=None)
-------------------------------------------------------------
n               grid size (n x n, toroidal boundary)
p               satisfaction threshold an agent needs to stay put
m               marriage parameter: per-step probability that an eligible agent is
                willing to marry, and cap on total marriages (m x initial population)
fixed_openness  if given, every agent (including newly created ones) uses this
                openness instead of a random value in {0.1, ..., 0.9}
"""

import numpy as np
from scipy.signal import correlate2d
from matplotlib.colors import ListedColormap
from Cell2D import Cell2D, draw_array  # provided in utils/ (see utils/README.md)

colors = [
    "white",  # 0 empty house: white
    "#D98282",  # 1 A: soft red
    "#7EA6CF",  # 2 B: soft blue
    "red",  # 3 AA: dark red
    "blue",  # 4 BB: dark blue
    "purple",  # 5 AB: purple (stands out more)
]

cmap = ListedColormap(colors)
class Person():

    def __init__(self, loc, group, person_id):
        """Represents a residential agent occupying one grid cell.
        Groups 1 and 2 represent single individuals.
        Groups 3, 4 and 5 represent two-person households.
        """

        # Give each person a unique ID
        # so it is easy later to record which households form and who lived next to whom
        self.id = person_id

        # Current location
        self.loc = tuple(loc)

        # Store this person's original group
        self.group = group

        # Each person independently draws a random openness
        # 0.1, 0.2, ..., 0.9
        self.openness = np.random.randint(1, 10) / 10

        # Happiness has not been calculated yet
        self.satisfaction = None
        self.happy = None

        # key = neighbour's id, value = number of consecutive rounds as neighbours
        self.neighbors = {}

    def get_info(self):
        """Return this person's current information."""

        return {
            "id": self.id,
            "loc": self.loc,
            "group": self.group,
            "openness": self.openness,
            "satisfaction": self.satisfaction,
            "happy": self.happy,
            "neighbors": self.neighbors.copy()
        }
        
    def update_happiness(self, satisfaction, p):
        """
        Update the current residential happiness.
        p: threshold on the fraction of similar neighbors (satisfaction threshold)
        """
        self.satisfaction = float(satisfaction)
        self.happy = bool(self.satisfaction >= p)

    def update_neighbors(self, current_neighbor_ids, advance_round=True):
        """
        current_neighbor_ids: IDs of the people actually living nearby in the current round
        """
        new_neighbors = {}

        for neighbor_id in current_neighbor_ids:
            # Existing neighbours keep their previous count; new neighbours start from 0
            rounds = self.neighbors.get(neighbor_id, 0)

            # Only increase the count when advancing to a new round
            if advance_round:
                rounds += 1

            new_neighbors[neighbor_id] = rounds

        self.neighbors = new_neighbors
    
def locs_where(condition):
    """Find cells where a logical array is True.

    condition: logical array

    returns: list of location tuples (identified by the row/column indices)
    """
    return list(zip(*np.nonzero(condition)))

class Schelling(Cell2D):
    """Represents a grid of Schelling agents."""

    options = dict(mode='same', boundary='wrap')

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.int8)

    def __init__(self, n, p, m = 0.1, fixed_openness=None):
        """Initializes the attributes.

        n: number of rows -- size of the grid
        p: threshold on the fraction of similar neighbors (satisfaction threshold)
        m: marriage
        """
        self.p = p
        self.fixed_openness = fixed_openness
        # 0 is empty house, 1 is agent labelled red, 2 is agent labelled blue
        choices = np.array([0, 1, 2], dtype=np.int8)
        probs = [0.1, 0.45, 0.45]

        # randomly initialize a grid of houses of n by, occupied by two kinds of agents (red and blue)
        # in roughly equal numbers. About 10% of the houses are empty.
        # in [0, 1, 2], o has 10% to be chosen, 1 and 2 each has 45% to be chosen.
        self.array = np.random.choice(choices, (n, n), p=probs) # this line randomly assigns 0 / 1 / 2 to every cell

        # The line above randomly assigned 0 (empty), 1 or 2 to every cell; now collect the information for each person
        # Dictionary: key = location, for example (3, 5) value = Person object
        # empty cells will NOT be stored here
        
        # Person IDs start from 0 for every new Schelling model
        self.next_person_id = 0

        self.people = {} # look up a person by their current location
        self.people_by_id = {} # look up a person by their ID

        # Loop over the whole grid to create every Person
        for row in range(n):
            for col in range(n):
                group = self.array[row, col] # only 0, 1 or 2 at this point
                # 0 is an empty house, so no Person is created
                if group != 0:
                    loc = (row, col) # create the Person at this location
                    person = Person(loc, group, self.next_person_id)
                    if self.fixed_openness is not None:
                        person.openness = float(self.fixed_openness)
                    self.next_person_id += 1
                    self.people[loc] = person # store the Person
                    self.people_by_id[person.id] = person # store the Person by ID

        # Marriage
        self.m = m
        self.marriage_limit = int(len(self.people) * m)
        self.marriages_done = 0

    def get_person_info(self, person_id):
        """Find a Person by ID and return that person's information."""
        person = self.people_by_id.get(person_id)
        if person is None: return None
        return person.get_info()
    
    def count_neighbors(self):
        """Surveys neighboring cells.

        returns: tuple of
            empty: True where cells are empty
            frac_red: fraction of red neighbors around each cell
            frac_blue: fraction of blue neighbors around each cell
            frac_same: fraction of neighbors with the same color
        """
        a = self.array
        empty = a == 0
        # AA counts as pure A, BB counts as pure B, AB is counted separately
        pure_a = (a == 1) | (a == 3)
        pure_b = (a == 2) | (a == 4)
        mixed = a == 5

        # correlate with the kernel, considering the 8 surronding neighbours
        # for agent labelled by red, count how many neighbours are also red agents
        # for agent labelled by blue, count how many neigbours are also blue neighbours
        num_a = correlate2d(pure_a, self.kernel, **self.options)
        num_b = correlate2d(pure_b, self.kernel, **self.options)
        num_ab = correlate2d(mixed, self.kernel, **self.options)
        
        # for each aget, count the total neighbours (no matter blue or red) it has
        num_neighbors = num_a + num_b + num_ab

        # compute fraction of similar neighbors (separately for A, B and AB neighbours)
        # Some cells may have no occupied neighbours, so num_neighbors can be 0.
        # Ignore the temporary division warning here.
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_a = num_a / num_neighbors
            frac_b = num_b / num_neighbors
            frac_ab = num_ab / num_neighbors

        # no neighbors is considered the same as no similar neighbors (if a cell has no occupied neighbours, its fractions are set to 0)
        # (this is an arbitrary choice for a rare event)

        frac_a[num_neighbors == 0] = 0
        frac_b[num_neighbors == 0] = 0
        frac_ab[num_neighbors == 0] = 0

        # fraction of neighbouring households of the same type as this cell
        frac_same = np.where(pure_a, frac_a, np.where(pure_b, frac_b, frac_ab))

        # where red is False could be either blue or empty
        # thus, one additional step:
        # for empty cells, frac_same is NaN
        # handle empty houses separately
        frac_same[empty] = np.nan

        # Copy frac_same first so satisfaction has the same grid shape
        satisfaction = frac_same.copy()
        # Each person uses their own openness to compute satisfaction
        # A: each A/AA and AB neighbour scores 1; each B/BB neighbour scores person.openness.
        # B: each B/BB and AB neighbour scores 1; each A/AA neighbour scores person.openness.
        # AA: each A/AA and AB neighbour scores 1; B/BB scores 0.
        # BB: each B/BB and AB neighbour scores 1; A/AA scores 0.
        # AB: each AB neighbour scores ab_weight (1.5); each pure A or pure B household scores person.openness.
        for loc, person in self.people.items():
            total = num_neighbors[loc]

            if total == 0:
                score = 0

            elif person.group == 1:  # A
                score = (num_a[loc] + num_ab[loc] + person.openness * num_b[loc]) / total

            elif person.group == 2:  # B
                score = (num_b[loc] + num_ab[loc] + person.openness * num_a[loc]) / total

            elif person.group == 3:  # AA
                score = (num_a[loc] + num_ab[loc]) / total

            elif person.group == 4:  # BB
                score = (num_b[loc] + num_ab[loc]) / total

            elif person.group == 5:  # AB
                ab_weight = 1.5
                score = (ab_weight * num_ab[loc] + person.openness * (num_a[loc] + num_b[loc])) / total

            else:
                raise ValueError("Unknown group")

            satisfaction[loc] = score

        return empty, frac_a, frac_b, frac_same, satisfaction

    def segregation(self):
        """Computes the average fraction of similar neighbors.
        returns: fraction of similar neighbors, averaged over cells
        Interpretation: a return value of 0.72 means that,
        on average, about 72% of each agent's neighbours are of the same type as itself.
        """
        # For quantitative analysis of segregation
        _, _, _, frac_same, _ = self.count_neighbors()
        return np.nanmean(frac_same)

    def update_happiness(self, satisfaction):
        """Update each Person's satisfaction and happy state."""
        # Update every person's satisfaction and happiness

        for loc, person in self.people.items():

            # satisfaction is how satisfied the person is at their current location
            score = satisfaction[loc]
            person.satisfaction = float(score)

            # update whether each person is happy
            person.happy = bool(score >= self.p)

    def update_neighbors(self, advance_round=True):
        """Update every Person's current neighbors."""

        # get the number of rows and columns of the grid
        rows, cols = self.array.shape

        for person in self.people.values():
            row, col = person.loc

            current_neighbor_ids = set() # stores the IDs of all of this person's current neighbours

            # check the 8 surrounding cells
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):

                    # do not count the person as their own neighbour
                    if dr == 0 and dc == 0:
                        continue

                    # modulo implements the toroidal boundary (boundary='wrap') used by the model
                    neighbor_loc = ((row + dr) % rows, (col + dc) % cols)

                    neighbor = self.people.get(neighbor_loc)

                    if neighbor is not None:
                        current_neighbor_ids.add(neighbor.id)

            person.update_neighbors(current_neighbor_ids, advance_round=advance_round)

    def find_candidates(self, person_id, required_rounds=3):
        """Find possible AA or BB pairs."""
        # list everyone that person_id could pair with
        candidates = []

        # find the current Person by ID only
        person = self.people_by_id.get(person_id)

        if person is None:
            return candidates

        # only singles can form a household
        if person.group != 1 and person.group != 2:
            return candidates

        # the person must be happy
        if person.happy is not True:
            return candidates
    
        # check all of this person's neighbours
        for neighbor_id in person.neighbors:

            rounds = person.neighbors[neighbor_id]

            # skip if they have not been neighbours for enough consecutive rounds
            if rounds < required_rounds:
                continue

            # find the neighbouring Person by ID
            neighbor = self.people_by_id.get(neighbor_id)

            # the neighbour must exist
            if neighbor is None:
                continue

            # the neighbour must also be single
            if neighbor.group != 1 and neighbor.group != 2:
                continue

            # the neighbour must also be happy
            if neighbor.happy is not True:
                continue

            candidates.append(neighbor.id)

        return candidates

    def find_all_candidate_pairs(self, required_rounds=3):
        """find_all non-conflicting pairs."""

        # first collect all candidates
        possible_pairs = []
        for person_id in self.people_by_id:
            candidates = self.find_candidates(person_id, required_rounds)

            for candidate_id in candidates:
                        # store each pair only once across the whole model
                        if person_id < candidate_id:
                            possible_pairs.append((person_id, candidate_id))

        return possible_pairs

    def choose_marriage_pairs(self, possible_pairs):
        # how many more marriages the model still allows
        remaining = self.marriage_limit - self.marriages_done
        if remaining <= 0:
            return []

        # find everyone involved in the candidate pairs
        candidate_ids = set() # collect the IDs from all pairs; a set removes duplicates
        for person_id, partner_id in possible_pairs:
            candidate_ids.add(person_id)
            candidate_ids.add(partner_id)

        # each person draws only once per round: willing to marry or not
        willing_ids = set()
        for person_id in candidate_ids:
            if np.random.random() < self.m: willing_ids.add(person_id)

        # keep only pairs where both people are willing this round
        willing_pairs = []
        for first_id, second_id in possible_pairs:
            if first_id in willing_ids and second_id in willing_ids:
                willing_pairs.append((first_id, second_id))

        marriage_pairs = []
        paired_ids = set()

        while len(willing_pairs) > 0 and len(marriage_pairs) < remaining:
            # compute the weight of each remaining pair
            weights = []

            for first_id, second_id in willing_pairs:
                first = self.people_by_id[first_id]
                second = self.people_by_id[second_id]

                if first.group == second.group:
                    weights.append(1)
                else:
                    weights.append(first.openness * second.openness)

            # convert the weights into selection probabilities
            probabilities = np.array(weights) / sum(weights)
            random_index = np.random.choice(len(willing_pairs), p=probabilities)

            first_id, second_id = willing_pairs[random_index]
            marriage_pairs.append((first_id, second_id))
            paired_ids.add(first_id)
            paired_ids.add(second_id)

            # once someone marries, remove every other pair that includes them
            remaining_pairs = []
            for person_id, partner_id in willing_pairs:
                if person_id not in paired_ids and partner_id not in paired_ids:
                    remaining_pairs.append((person_id, partner_id))

            willing_pairs = remaining_pairs

        return marriage_pairs

    def complete_marriages(self, marriage_pairs):
        """Carry out the marriages selected this round and return how many were completed."""
        completed = 0

        for first_id, second_id in marriage_pairs:
            if self.marriages_done >= self.marriage_limit:
                break

            first = self.people_by_id.get(first_id)
            second = self.people_by_id.get(second_id)

            # both people must still exist and still be single
            if first is None or second is None:
                continue
            if first.group not in (1, 2) or second.group not in (1, 2):
                continue

            first_loc = first.loc
            second_loc = second.loc

            # decide the household type and the type of the new single
            if first.group == 1 and second.group == 1:
                married_group = 3          # AA
                child_group = 1            # A
            elif first.group == 2 and second.group == 2:
                married_group = 4          # BB
                child_group = 2            # B
            else:
                married_group = 5          # AB
                if np.random.random() < 0.5:
                    child_group = 1        # A
                else:
                    child_group = 2        # B

            # randomly decide which of the two original houses the household occupies
            if np.random.random() < 0.5:
                married_loc = first_loc
                child_loc = second_loc
            else:
                married_loc = second_loc
                child_loc = first_loc

            married = Person(married_loc, married_group, self.next_person_id)
            self.next_person_id += 1

            child = Person(child_loc, child_group, self.next_person_id)
            if self.fixed_openness is not None:
                child.openness = float(self.fixed_openness)
            self.next_person_id += 1

            # record which two former Persons formed this household
            married.member_ids = (first_id, second_id)

            if married_group == 5:
                married.openness = min(first.openness, second.openness)
            else:
                married.openness = None     # AA and BB do not use openness

            # remove the two former Persons
            del self.people[first_loc]
            del self.people[second_loc]
            del self.people_by_id[first_id]
            del self.people_by_id[second_id]

            # register the new Persons at the two original locations
            self.people[married_loc] = married
            self.people[child_loc] = child
            self.people_by_id[married.id] = married
            self.people_by_id[child.id] = child

            self.array[married_loc] = married_group
            self.array[child_loc] = child_group

            self.marriages_done += 1
            completed += 1

        # after all marriages are processed, refresh neighbours without advancing the round count
        if completed > 0: self.update_neighbors(advance_round=False)

        return completed

    def step(self):
        """Executes one time step.
        returns: fraction of similar neighbors, averaged over cells
        """
        a = self.array
        empty, _, _, frac_same, satisfaction = self.count_neighbors()

        # Update Person objects
        self.update_happiness(satisfaction)

        # find the unhappy cells (ignore NaN in frac_same)
        unhappy_locs = [loc for loc, person in self.people.items() if person.happy is False]


        # find the empty cells: the list of (row, column) index tuples for empty location
        empty_locs = locs_where(empty)
        num_empty = len(empty_locs)

        # If there are no empty houses, nobody can move
        if num_empty == 0:
            unhappy_locs = []

        # shuffle the unhappy cells
        # shuffling avoids always moving in a fixed row 0 → row 1 → row 2 order, which would add an artificial directional bias
        if len(unhappy_locs):
            np.random.shuffle(unhappy_locs)

        # for each unhappy cell, choose a random destination
        num_empty = np.sum(empty)

        for source in unhappy_locs:
            i = np.random.randint(num_empty) # pick a random empty house
            dest = empty_locs[i] # the house this person will move into

            # find the Person at source
            # and remove the old location's entry from the dictionary
            # pop returns the value and removes that entry from the dictionary
            person = self.people.pop(source)

            # move
            a[dest] = person.group # unhappy agent moves to the empty location, state value updates from 0 to the agent's group
            a[source] = 0 # the unhappy agent's location becomes empty after it moves away
            person.loc = tuple(dest) # update person's location in class

            self.people[dest] = person # update the people dictionary of this model
            empty_locs[i] = source # update the empty location list to include the location of unhappy agent that just moved away
            
        # check that the number of empty cells remains the same
        assert num_empty == np.sum(a==0)
        
        # recompute happiness after moving
        _, _, _, updated_frac_same, updated_satisfaction = self.count_neighbors()
        self.update_happiness(updated_satisfaction)

        # update neighbours based on the final locations after moving
        self.update_neighbors()

        # The marriage process added in our extension
        if self.marriages_done < self.marriage_limit:
            possible_pairs = self.find_all_candidate_pairs()
            marriage_pairs = self.choose_marriage_pairs(possible_pairs)

            if len(marriage_pairs) > 0:
                completed = self.complete_marriages(marriage_pairs)

                if completed > 0:
                    _, _, _, updated_frac_same, updated_satisfaction = self.count_neighbors()
                    self.update_happiness(updated_satisfaction)

        # return the average fraction of similar neighbors
        return np.nanmean(updated_frac_same)

    def draw(self):
        """Draws the cells."""
        return draw_array(self.array, cmap=cmap, vmin=0, vmax=5, alpha=1)
