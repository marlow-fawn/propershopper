
import json
import socket
import random
from copy import deepcopy

import pandas as pd
import pathfind.graph.transform

from enums.direction import Direction
from final_proj.util import get_geometry
from helper import project_collision
from util import *
from final_proj.fast_high_level_astar import *
from box_regions import *

# todo: update with Helen's method for making interaction areas?
def populate_locs(observation):
    # add interaction areas to objects in the observation
    obs_with_boxes = add_interact_boxes_to_obs(obs=observation)
    locs: dict = {}

    for idx, obj in enumerate(obs_with_boxes['registers']):
        geometry = get_geometry(obj)
        geometry['position'][0] += geometry['width'] + 1
        geometry['interact_boxes'] = obj['interact_boxes']
        locs[f'register {idx}'] = geometry

    for idx, obj in enumerate(obs_with_boxes['cartReturns']):
        geometry = get_geometry(obj)
        geometry['position'][1] -= 1
        geometry['interact_boxes'] = obj['interact_boxes']
        locs[f'cartReturn {idx}'] = geometry

    for idx, obj in enumerate(obs_with_boxes['basketReturns']):
        geometry = get_geometry(obj)
        geometry['position'][1] -= 1
        geometry['interact_boxes'] = obj['interact_boxes']
        locs[f'basketReturn {idx}'] = geometry

    for obj in obs_with_boxes['counters']:
        geometry = get_geometry(obj)
        geometry['position'][0] -= 1
        geometry['interact_boxes'] = obj['interact_boxes']
        locs[obj['food']] = geometry

    for obj in obs_with_boxes['shelves']:
        geometry = get_geometry(obj)
        geometry['position'][1] += geometry['height'] + 1
        geometry['interact_boxes'] = obj['interact_boxes']
        locs[obj['food']] = geometry
    
    for idx, obj in enumerate(obs_with_boxes['carts']):
        geometry = get_geometry(obj)
        geometry['position'][1] += geometry['height'] + 1
        geometry['interact_boxes'] = obj['interact_boxes']
        locs['cart {idx}'] = geometry
    
    for idx, obj in enumerate(obs_with_boxes['baskets']):
        geometry = get_geometry(obj)
        geometry['position'][1] += geometry['height'] + 1
        geometry['interact_boxes'] = obj['interact_boxes']
        locs['basket {idx}'] = geometry

    return locs


class Agent:

    def __init__(self, conn, agent_id, env):
        self.socket = conn
        self.agent_id = agent_id
        self.env = env
        self.shopping_list = []
        self.list_quant:list = env['observation']['players'][self.agent_id]['list_quant']
        for item, quant in zip(env['observation']['players'][self.agent_id]['shopping_list'], self.list_quant):
            self.shopping_list += [item] * quant
        self.shopping_list.reverse() # reverse the list so we don't go for the same item at the same time as other agents
        self.goal = ""
        self.goal_status = None # mainly for navigation and replanning
        self.done = False
        self.container_id = -1
        self.container_type = ''
        self.holding_container = False
        self.holding_food = None
        self.planner = HighLevelPlanner(socket_game=conn, env=env)

    def transition(self):
        self.execute(action='NOP') # this updates self.env
        if self.done:  # If we've left the store
            self.goal_status = 'pending'
            self.execute("NOP")  # Do nothing

        elif self.container_id == -1 or self.goal == 'cart' or self.goal == 'basket':  # If we don't have a container
            self.goal_status = 'pending'
            self.get_container()  # Get one!

        elif self.goal == "":  # If we currently don't have a goal
            if not self.shopping_list:  # Check if there's anything left on our list
                self.goal_status = 'pending'
                self.exit()  # Leave the store (includes checkout), might need to return basket
            else:  # We've still got something on our shopping list
                item = self.strategically_choose_from_shopping_list(self.shopping_list)
               
                self.goal = item  # Set our goal to the next item on our list
                self.goal_status = 'pending'
                self.get_item()  # Go to our goal
        elif self.goal == 'add_to_container':
            self.goal_status = 'pending'
            self.add_to_container()
        elif self.goal == 'put_back_food':
            self.goal_status = 'pending'
            self.put_back_food()
        else: # this shouldn't happen, just exit
            self.goal_status = 'pending'
            self.exit()

    
    def get_item(self):
        """get item

        Args:
            item (str): an item on the shopping list
        """
        # precondition check
        if not self.holding_container:# must be holding a container
            self.goal_status = 'failure'
            self.goal = self.container_type # otherwise change goal to get container first
            return 
        
        print(f"Agent {self.agent_id} going to {self.goal} with {self.container_type}")
        # go to the item and get it. Look at the implementation of `get_container` for reference. target item is stored in self.goal
        self.goto(self.goal) # go to food item
        self.execute('TOGGLE_CART') # let go of container
        self.goto(self.goal) # go face the food item
        self.execute('INTERACT')
        self.execute('INTERACT')
        # set postconditions
        self.holding_food = self.env['observation']['players'][self.agent_id]['holding_food']
        if self.holding_food is not None:
            self.holding_container = False # sanity check: can't hold both food and container
        if self.holding_food == self.goal:# successfully got item
            self.goal_status = 'success'
            self.goal = "add_to_container" # add item to container
            return
        elif self.holding_food is None: # we didn't get the food
            self.goal_status = 'failure'
            return
        elif self.holding_food != self.goal: # we got the wrong food
            self.goal_status = 'failure'
            self.goal = 'put_back_food'
            return
    

    def put_back_food(self):
        """Put the food we are holding back on its shelf. Used when accidentally grabbed the wrong food

        Returns:
            None
        """
        # preconditions check
        if self.holding_food is None: # not holding anything
            self.goal_status = 'success'
            return
        
        self.goto(goal=self.holding_food) # go to the shelf for the food we are holding
        self.execute('INTERACT') # interact with the shelf to put the food back

        # postconditions check
        if self.holding_food is None: # no longer holding food
            self.goal_status = 'success'
            self.goal = ''
            return
        else:
            self.goal_status = 'failure'
            self.goal = 'put_back_food'
            return

    
    # Agent retrieves a container
    def get_container(self):
        if sum(self.list_quant) <= 6:
            self.update_container(container='basket')
            print(f"Agent {self.agent_id} getting a basket")
            if self.container_id == -1: # has never gotten a container
                self.goal = 'basketReturn 0'
                self.goto(goal='basketReturn 0', is_item=True)
                self.execute('INTERACT')
                self.execute('INTERACT')
            else:
                self.goal = 'basket'# we have gotten a basket before, it's somewhere in the environment
                self.goto(goal=f'basket {self.container_id}', is_item=True)
                self.execute('TOGGLE_CART')
            self.update_container('basket')
            self.holding_container = True #we have to make this assumption, it's not reflected in the env
        else:
            print(f"Agent {self.agent_id} getting a cart")
            self.update_container(container='cart')
            if self.container_id == -1: # has never gotten a container
                self.goal = 'cartReturn 0'
                self.goto(goal='cartReturn 0', is_item=True)
                self.execute('INTERACT')
                self.execute('INTERACT')
            else:
                self.goal = 'cart'
                self.goto(goal=f'cart {self.container_id}', is_item=True)
                self.execute('TOGGLE_CART')
            self.update_container('cart')
        self.goal = ""
        self.goal_status = 'success' if self.holding_container else 'failure'

    def strategically_choose_from_shopping_list(self, shopping_list):
        """Strategically choose an item from the shopping list

        Args:
            shopping_list (list): shopping list
        """
        # pick the least crowded
        min_crowdedness = 1.0
        self.execute("NOP") # make sure to observe where other players are for crowdedness calculations
        for item in shopping_list:
            crowdedness = calculate_crowdedness_factor(self.agent_id, locs[item]['interact_boxes']['SOUTH_BOX'], self.env['observation'])
            if crowdedness < min_crowdedness:
                best = item
                min_crowdedness = crowdedness
        return best
    
    def update_container(self, container='basket'):
        """Check if we are responsible for any `container` and update container related status. Either a cart or a basket

        Args:
            container (_type_): either a cart or a basket
        """
        if self.env['observation']['players'][self.agent_id]['curr_cart'] != -1:#currently holding a cart
            self.container_type = 'cart'
            self.container_id = self.env['observation']['players'][self.agent_id]['curr_cart']
            self.holding_container = True
            return
        for i, c in enumerate(self.env['observation'][container+'s']):
            if c['owner'] == self.agent_id:
                self.container_id = i
                self.container_type = container
                return
    
    
    def goto(self, goal:list|tuple|str, is_item=True):
        """go to the goal, either a (x, y) of a string such as 'basket', 'register'

        Args:
            goal (list | tuple | str): (x, y) or strings such as 'strawberry', 'basket'
            is_item (bool, optional): if the goal is an item. Set False for (x, y). Defaults to True.
        """
        if goal is None:
            goal = self.goal

        if is_item:#goal is an item in the env, not a (x, y)
            populate_locs(self.env['observation'])
            if goal in ('cartReturn 0', 'cartReturn 1', 'basketReturn 0', 'register 0', 'register 1'): # access these from the North
                interact_box = locs[goal]['interact_boxes']['NORTH_BOX']
                goal = self.interact_box_to_goal_location(box=interact_box)
            elif goal in ('cart', 'basket'):
                if self.container_type != goal:# current container and goal doesn't match, get current container instead
                    self.goto(goal=self.container_type)
                    return
                elif self.holding_container:#already holding container
                    self.goal_status = 'success' if self.container_type == goal else 'failure'
                    return
                else:
                    interact_boxes:dict = locs[f'{goal} {self.container_id}']
                    if goal == 'cart':
                        interact_box = list(interact_boxes.values())[0]#cart only has one interact box, go to that interact box
                        goal= self.interact_box_to_goal_location(box=interact_box)
                    else:
                        interact_box = locs[f'{goal} {self.container_id}']['interact_boxes']['SOUTH_BOX']
                        goal= self.interact_box_to_goal_location(box=interact_box)

            else:# access everything else from the SOUTH
                interact_box = locs[goal]['interact_boxes']['SOUTH_BOX']
                goal = self.interact_box_to_goal_location(box=interact_box)

        print(f"Agent {self.agent_id} planning a path to {goal} with container {self.container_type}")
        path = self.planner.astar(
            player_id=self.agent_id,
            start=self.env['observation']['players'][self.agent_id]['position'],
            goal=goal,
            obs=self.env['observation']
        )
        print(f"Agent {self.agent_id} going to location {goal} with container {self.container_type}")
        
        for box_region in path:
            if box_region.name == "W_walkway" or box_region.name == "E_walkway":
                self.reactive_nav(goal=box_region, align_x_first=True, is_box=True) # to navigate to a walkway, align x first
            else:
                self.reactive_nav(goal=box_region, align_x_first=False, is_box=True) # to navigate to any other region, align y first

            if self.goal_status == 'failure': # didn't get to goal
                self.transition()
                return # let the transition method replan

        # we should now be in the same region as the goal (x, y) location
        if is_item:
            self.reactive_nav(goal=interact_box, align_x_first=True, is_box=True)
        else:
            self.reactive_nav(goal=goal, align_x_first=True, is_box=False)
        
        if self.goal_status == 'failure': # didn't get to goal
            self.transition()
            return # replan
        
        self.goal_status = 'success'
        return

    
    def interact_box_to_goal_location(self, box:dict) -> tuple[float, float]:
        """Given an interact box, determine which goal location within the box to aim for

        Args:
            box (dict): interact box

        Returns:
            tuple[float, float]: the goal location
        """
        player_needs_to_face = box['player_needs_to_face']
        if player_needs_to_face == Direction.SOUTH:
            top_left = (box['westmost'],box['northmost'])
            return top_left
        elif player_needs_to_face == Direction.NORTH:
            bot_left = (box['westmost'],box['southmost'])
            return bot_left
        elif player_needs_to_face == Direction.WEST:
            bot_right = (box['eastmost'],box['southmost'])
            return bot_right
        else:
            bot_left = (box['westmost'],box['southmost'])
            return bot_left
    
    
    
    def reactive_nav(self, goal, align_x_first=False, is_box=False):
        """Purely reactie navigation

        Args:
            goal (_type_): (x, y) or interact_box
        """
        ##############################################
        ## The old reactive navigation code is below##
        ##############################################
        target = "x" if align_x_first else "y" # have to align y first for navigation from one region to another
        reached_x = False
        reached_y = False
        stuck = 0 # stuck for timestep
       
        while True:
            player = self.env['observation']['players'][self.agent_id]
            if self.holding_container and self.container_type == 'cart': # change player's shape to account for cart
                cart = self.env['observation']['carts'][self.container_id]
                player_cart = deepcopy(player)
                if player_cart['direction'] == Direction.NORTH.name:
                    player_cart['position'][1] -= cart['height'] # need to move the y of the player to the upper left of cart + player
                    player_cart['height'] += cart['height']
                elif player_cart['direction'] == Direction.SOUTH.name:
                    player_cart['height'] += cart['height']
                elif player_cart['direction'] == Direction.WEST.name:
                    player_cart['position'][0] -= cart['width'] # need to move the x of the player to the upper left of cart + player
                    player_cart['width'] += cart['width']
                else:
                    player_cart['width'][0] += cart['width'] 

            if is_box:
                if isinstance(goal, BoxRegion):# it's a box region
                    if goal.contains(player['position']):
                        break
                    goal_loc = goal.closest(point=player['position'])
                    x_dist = player['position'][0] - goal_loc[0]
                    y_dist = player['position'][1] - goal_loc[1]
                else: # it's an interaction box
                    goal_loc = self.interact_box_to_goal_location(box=goal)
                    x_dist = player['position'][0] - goal_loc[0]
                    y_dist = player['position'][1] - goal_loc[1]
                    if not (self.holding_container and self.container_type == 'cart') and can_interact_in_box(player=player, interact_box=goal):
                        break
                    elif (self.holding_container and self.container_type == 'cart') and loc_in_box(loc=player['position'], box=goal): # we are holding a cart, it's good enough to be in the interaction box. We don't have to face the right direction
                        break
            else:
                x_dist = player['position'][0] - goal[0]
                y_dist = player['position'][1] - goal[1]

            if abs(x_dist) < STEP:
                reached_x = True
            if abs(y_dist) < STEP:
                reached_y = True
            if reached_x and reached_y:
                if not is_box:
                    break
                else: 
                    if not isinstance(goal, BoxRegion) and not (self.holding_container and self.container_type == 'cart'): # it's an interact box, we have to face the right direction
                        self.execute(goal['player_needs_to_face'].name)
                        reached_x = False # reset because we could have moved due to stochasticity
                        reached_y = False # reset because we could have moved due to stochasticity
                    else:
                        break

            if target == "x":
                if x_dist < -STEP:
                    command = Direction.EAST
                elif x_dist > STEP:
                    command = Direction.WEST
                else:
                    reached_y = False
                    target = "y"
                    continue
            else:
                if y_dist < -STEP:
                    command = Direction.SOUTH
                elif y_dist > STEP:
                    command = Direction.NORTH
                else:
                    reached_x = False
                    target = "x"
                    continue
            original_command = command

            if self.holding_container and self.container_type == 'basket':
                player['curr_basket'] = self.container_id # a hack. curr_basket isn't reflected in the env unlike curr_cart
            if self.holding_container and self.container_type == 'cart':
                player = player_cart # treat player and cart as one bigger object when projecting collision
            while project_collision(player, self.env, command, dist=STEP):
                command = Direction(self._ninety_degrees(dir=command)) # take the 90 degrees action instead
                stuck += 1
                
                if stuck >= 200: # been stuck for eternity, replan
                    self.goal_status = 'failure'
                    return # let the goto method replan
                elif stuck >= 20:#been stuck for too long, it's probably a static corner some other agent created, F it, try a random action 
                    command = random.choice([dir for dir in Direction if (dir != Direction.NONE and dir != original_command)])
                    if not project_collision(player, self.env, command, dist=STEP):
                        stuck = 0
                        # once no longer stuck, try switching alignment target to avoid walking back into the corner
                        if target == "x":
                            target = "y"
                        else:
                            target = "x"
                        break
                reached_x = False
                reached_y = False
            
            
            if player['direction'] == command.value:
                self.execute(action=command.name)# execute once if already facing that direction
            else: # execute twice if need to turn in that direction first
                self.execute(action=command.name)
                self.execute(action=command.name)
  
    def _opposite_dir(self, dir:Direction) -> Direction:
        """Turn 180 degrees with respect to the given 

        Args:
            command (Direction): the direction command whose ninety degree direction we want to find
        Returns:
            returns the 90 degrees direction
        """
        if dir == Direction.NORTH:
            return Direction.SOUTH
        if dir == Direction.SOUTH:
            return Direction.NORTH
        if dir == Direction.EAST:
            return Direction.WEST
        else:
            return Direction.EAST

    def _ninety_degrees(self, dir:Direction) -> Direction:
        """Find the command that's 90 degrees clockwise with respect to the given dir

        Args:
            command (Direction): the direction command whose ninety degree direction we want to find
        Returns:
            returns the 90 degrees direction
        """
        turned_dir = Direction((dir.value + 2) % 5)
        if turned_dir == Direction.NONE:
            return Direction.SOUTH
        return turned_dir

    
    
    def step(self, step_location:list|tuple, player_id:int, backtrack:list):
        """Keep locally adjusting and stepping in the right direction so that `player` ends up `close_enough` to `step_location`. `step_location` should be only one step away

        Args:
            step_location (list | tuple): a (x, y) that is assumed to be one step away from the player's current location
            player_id (int): the player id for the player that needs to be at `step_location`
            backtrack (list): a list of locations visited by the player
        """
        goal_x, goal_y = step_location
        player_x, player_y = self.env['observation']["players"][player_id]['position']
        while not self.planner.is_close_enough(current=self.env['observation']["players"][player_id]['position'], goal=step_location, tolerance=LOCATION_TOLERANCE, is_item=False):#deals with stochasticity: keep locally adjusting to the right location until it's close enough
            # compare previous position with current position to determine if a location needs to be saved in the player's backtracking trace 
            prev_x = player_x
            prev_y = player_y
            player_x, player_y = self.env['observation']["players"][player_id]['position']
            if player_x != prev_x or player_y != prev_y:#player has moved, record its prev position for potential backtracking
                backtrack.append((prev_x, prev_y))

            if manhattan_distance(self.env['observation']["players"][player_id]['position'], step_location) >= BACKTRACK_TOLERANCE:#player has wandered too far due to stochasticity, there could be an object between the player and the goal `step_location` now. The player needs to backtrack to the starting location, otherwise it could be banging its head against the object forever
                self.step(step_location=backtrack[-1], player_id=player_id, backtrack=backtrack[:-1])
                del backtrack[-1]
            elif player_x < goal_x and abs(player_x - goal_x) >= LOCATION_TOLERANCE:# player should go EAST
                #self.execute(Direction.EAST.name)
                self.reactive_nav(goal=step_location, is_box=False)
            elif player_x > goal_x and abs(player_x - goal_x) >= LOCATION_TOLERANCE:#player should go WEST
                #self.execute(Direction.WEST.name)
                self.reactive_nav(goal=step_location, is_box=False)
            elif player_y < goal_y and abs(player_y - goal_y) >= LOCATION_TOLERANCE:#player should go SOUTH
                #self.execute(Direction.SOUTH.name)
                self.reactive_nav(goal=step_location, is_box=False)
            elif player_y > goal_y and abs(player_y - goal_y) >= LOCATION_TOLERANCE:#player should go NORTH
                #self.execute(Direction.NORTH.name)
                self.reactive_nav(goal=step_location, is_box=False)

    # Agent picks up an item and adds it to the cart
    def add_to_container(self):
        print(f"Agent {self.agent_id} adding an item to the {self.container_type}")
        if self.holding_food and self.container_id == -1: # we have never gotten a container but we are holding a food
            self.goal = 'put_back_food'
            self.goal_status = 'failure'
            return
            
        if self.holding_food is None or self.container_id == -1: # we are not holding a food or we don't have a container
            self.goal_status = 'failure'
            self.goal = ''
            return
        
        self.goto(goal=f"{self.container_type} {self.container_id}") # we are holding food and we have a container, go to our container
        # take a look at the contents of the container before we try to put the food in
        num_contents = len(self.env['observation'][self.container_type][self.container_id]['contents'])
        self.execute('INTERACT') # put food in container
        # check if we successfully put the item into the container
        if len(self.env['observation'][self.container_type][self.container_id]['contents']) > num_contents:
            self.holding_food = None
            self.goal_status = 'success'
            # remove item from list
            self.shopping_list.remove(self.holding_food) 
            self.goal = self.container_type
            return
        else:
            self.goal_status = 'failure'
            return

    # goes to the register, checks out, and leaves
    # assumes that the agent is holding a cart or basket
    def exit(self):
        print(f"Agent {self.agent_id} exiting")
        self.goto(goal='register 1', is_item=True)

        action_arr = ["EAST", "TOGGLE_CART", "NORTH", "NORTH",
                      "INTERACT", "INTERACT", "SOUTH", "SOUTH",
                      "EAST", "TOGGLE_CART"]
        for a in action_arr:
            self.execute(a)

        exit_pos = [-0.8, 15.6]
        self.goto(exit_pos, is_item=False)#upper exit

        self.done = True

    # Reads the shopping list
    def init_list(self):
        print(f"Agent {self.agent_id} reading list")
        shopping_list = []
        self.execute("NOP")
        shopping_list += self.env['observation']["players"][self.agent_id]["shopping_list"].copy()
        return shopping_list

    # Given an action, executes it for this agent
    def execute(self, action):
        action = f"{self.agent_id} {action}"
        self.socket.send(str.encode(action))  # send action to env
        output = recv_socket_data(self.socket)  # get observation from env
        self.env = json.loads(output)
        # update all relevant observations
        self.holding_food = self.env['observation']['players'][self.agent_id]['holding_food']
        self.update_container('basket')
        self.update_container('cart')


if __name__ == "__main__":

    action_commands = ['NOP', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT']

    print("action_commands: ", action_commands)

    # Connect to Supermarket
    HOST = '127.0.0.1'
    PORT = 9000
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_game.connect((HOST, PORT))
    sock_game.send(str.encode("0 RESET"))  # reset the game
    state = recv_socket_data(sock_game)
    sock_game.send(str.encode("0 NOP"))  # send action to env
    output = recv_socket_data(sock_game)  # get observation from env
    output = json.loads(output)
    locs = populate_locs(output['observation'])
    agents = [Agent(sock_game, 0, env=output)]
    # agents = [Agent(sock_game, 0), Agent(sock_game, 1), Agent(sock_game, 2)]
    while True:
        for agent in agents:
            agent.transition()
