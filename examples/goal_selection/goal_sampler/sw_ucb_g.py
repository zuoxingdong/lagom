import numpy as np

from collections import deque
from collections import OrderedDict

from .base import BaseGoalSampler
from utils import get_optimal_steps

from lagom.runner import Runner


class SWUCBgGoalSampler(BaseGoalSampler):
    def __init__(self, runner, config):
        super().__init__(runner, config)
        
        # Counter for iterations
        self.t = 0
        
        # Sliding window
        self.h = 200
        
        # Partition of the goal space into groups
        self.groups = self._partition_goal_space()
        self.num_groups = len(self.groups)
        
        # Sliding-window queue of selected information
        self.queue = deque([], maxlen=self.h)  # queue with size as sliding window
        
        # Initialize goal value estimate
        self.infos = OrderedDict()
        for i, group in enumerate(self.groups):  # iterate over groups of goals
            d = OrderedDict()
            for goal in group:  # iterate over goals
                d[('Goal', tuple(goal))] = {'N': 0, 'Q': deque([], maxlen=self.h)}
            # Add to dictionary
            self.infos[('Group', i)] = d
            
        # hyperparams from SWUCB-g paper
        self.D1 = 0.1
        self.D2 = 2
        self.gamma1 = 2
        self.gamma2 = 0.5
        self.alphas = [len(group)+5 for group in self.groups]
            
    def _partition_goal_space(self):
        area1 = [[5, 1], [5, 2], [5, 3],
                 [6, 1], [6, 2], [6, 3]]
        area2 = [[1, 4], [1, 5], 
                 [2, 4], [2, 5], 
                 [3, 4], [3, 5],
                 [4, 4], [4, 5],
                 [5, 4], [5, 5],
                 [6, 4], [6, 5]]
        area3 = [[1, 1], [1, 2], [1, 3], 
                 [2, 1], [2, 2], [2, 3]]
        
        return [area1, area2, area3]
        
    def sample(self):
        # Group selection
        group_id = self._group_selection()
        # Goal selection
        goal = self._goal_selection(group_id)
        
        # Record sampled group and sampled goal for update()
        self.sampled_group_id = group_id
        self.sampled_goal = tuple(goal)
        
        # Increment the counter
        self.t += 1
        
        return list(goal)
    
    def update(self, reward):
        """
        Update the dictionary of reward information of the group and the goal
        
        Args:
            reward (float): reward to the bandit
        """
        # Update the sliding-window queue of selected information
        if len(self.queue) == self.queue.maxlen:
            pop_item = self.queue.popleft()
            # Update dictionary
            info = self.infos[('Group', pop_item['Group'])][('Goal', tuple(pop_item['Goal']))]
            info['N'] -= 1
            R = info['Q'].popleft()
            assert R == pop_item['R']  # check if the correct value to be poped out
        
        # Record new information to the queue
        self.queue.append({'Group': self.sampled_group_id, 'Goal': self.sampled_goal, 'R': reward})
        # Update dictionary
        info = self.infos[('Group', self.sampled_group_id)][('Goal', tuple(self.sampled_goal))]
        info['N'] += 1
        info['Q'].append(reward)
        
        # Clean the recorded sampled group id and goal
        self.sampled_group_id = None
        self.sampled_goal = None
    
    def _group_selection(self):
        """
        Group selection
        
        Returns:
            group_id (int): the index of the selected group
        """
        if self.t < self.num_groups:  # initially linearly choose the group
            group_id = self.t
        else:
            # TODO: two ways, either max of max Q, or max of average Q within the group
            value_groups = []
            for i in range(self.num_groups):
                goal, Q = self._max_goal(i)
                uncertainty = self._group_uncertainty(i, original=True)
                value_groups.append(Q + uncertainty)
                
            group_id = np.argmax(value_groups)
        
        return group_id
    
    def _goal_selection(self, group_id):
        """
        Goal selection within the group
        
        Args:
            group_id (int): the index of the selected group
            
        Returns:
            goal (tuple): selected goal
        """
        if self.t < self.num_groups:  # initially uniformly sample the goal within the group
            goals = self.groups[group_id]
            idx = np.random.choice(range(len(goals)))
            goal = goals[idx]
        else:
            # TODO: two ways, either with or without uncertainty
            info = self.infos[('Group', group_id)]
            value_goals = []
            for goal in self.groups[group_id]:
                if len(info[('Goal', tuple(goal))]['Q']) == 0:  # empty
                    Q = 0
                else:
                    Q = np.mean(info[('Goal', tuple(goal))]['Q'])
                uncertainty = self._goal_uncertainty(group_id, goal)
                value_goals.append(Q + uncertainty)
            
            idx = np.argmax(value_goals)
            goal = self.groups[group_id][idx]
        
        return goal
    
    def _max_goal(self, group_id):
        """
        Returns the goal with max Q within the group
        
        Args:
            group_id (int): ID of the group
            
        Returns:
            goal (tuple or list): goal with max Q
            Q (float): the Q associated with the goal
        """
        info = self.infos[('Group', group_id)]
        goals = []
        Qs = []
        for g, g_info in info.items():
            goals.append(g[1])
            if len(g_info['Q']) == 0:  # empty
                Q_value = 0
            else:
                Q_value = np.mean(g_info['Q'])
            Qs.append(Q_value)
            
        idx = np.argmax(Qs)
        goal = goals[idx]
        Q = Qs[idx]
        
        return goal, Q
    
    def _group_uncertainty(self, group_id, original=True):
        """
        Calculate uncertainty term for the group
        
        Args:
            group_id (int): ID of the group
            original (bool): If True, then use the original paper setting
            
        Returns:
            uncertainty (float): uncertainty term of the given group
        """
        if original:
            c = self.D2*(self.D1)**(-self.gamma2/self.gamma1)
            exponent = self.gamma2/(2*self.gamma1)
            alpha = self.alphas[group_id]
            ln = np.log(np.min([self.t, self.h]))
            N = np.sum([info['N'] for info in self.infos[('Group', group_id)].values()])
            
            ### TODO: here is a limitation, because it will try each goal at least once, intractable in large space
            ### Here also has to consider size of sliding window to avoid constant existence of zero N
            if N == 0:
                uncertainty = np.finfo(np.float32).max  # encourage selecting this goal maximally
            else:
                uncertainty = c*(alpha*(ln/N))**exponent
        else:
            raise NotImplementedError
            
        return uncertainty
    
    def _goal_uncertainty(self, group_id, goal):
        """
        Calculate uncertainty term for the goal within the group
        
        Args:
            group_id (int): ID of the group
            goal (tuple or list): sampled goal
            
        Returns:
            uncertainty (float): uncertainty term of the given group
        """
        c = 0.5
        ln = np.log(np.min([self.t, self.h]))
        N = self.infos[('Group', group_id)][('Goal', tuple(goal))]['N']
        
        ### TODO: here is a limitation, because it will try each goal at least once, intractable in large space
        ### Here also has to consider size of sliding window to avoid constant existence of zero N
        if N == 0:
            uncertainty = np.finfo(np.float32).max  # encourage selecting this goal maximally
        else:
            uncertainty = c*np.sqrt(ln/N)
        
        return uncertainty
        
    def _calculate_reward(self, agent_old, agent_new, env, config, goal):
        """
        Calculate the learning progress signal as reward
        
        Args:
            agent_old (Agent): deepcopy of agent before training
            agent_new (Agent): agent after training
            env (Env): environment
            config (Config): configurations
            goal (list or tuple): trained goal
            
        Returns:
            reward (float): learning progress reward signal
        """
        # Ensure the sampled goal is consistent 
        assert tuple(self.sampled_goal) == tuple(goal)
        
        # Set goal in the environment
        env.get_source_env().goal_states = [goal]
        
        # Set max time steps as optimal trajectories (consistent with A* solution)
        if config['use_optimal_T']:
            T = get_optimal_steps(env)
        else:
            T = config['T']
            
        num_epi = 15
        
        # Create runners
        runner_old = Runner(agent_old, env, config['gamma'])
        runner_new = Runner(agent_new, env, config['gamma'])
        
        # Calculate value estimate
        D_old = runner_old.run(T, num_epi)
        D_new = runner_new.run(T, num_epi)
        r_old = np.mean([np.sum(d['rewards']) for d in D_old])
        r_new = np.mean([np.sum(d['rewards']) for d in D_new])
        
        print(f'########### {r_old}, {r_new}')
        
        reward = r_new - r_old
        
        return reward