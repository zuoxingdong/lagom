from itertools import chain

from .trajectory import Trajectory

from .base_history import BaseHistory


class Segment(BaseHistory):
    r"""Define a segment of successive transitions from one or multiple episodes. 
    
    .. note::
    
        In general, a segment could describe transitions from a single episode or as a rolling segments of
        transitions from multiple episodes. Let :math:`s_{i, t}` be the state in episode :math:`i` at time
        step :math:`t` and the state is a terminal state when :math:`t=T`, we describe the possible use cases
        for a segment with length :math:`5` in the following:
        
        * Part of a single episode:
          
          .. math::
              s_{1, 1}, s_{1, 2}, s_{1, 3}, s_{1, 4}, s_{1, 5}, s_{1, 6}
              
          The corresponding ``done`` in the transitions should be: 
          
          .. math::
              [\text{False}, \text{False}, \text{False}, \text{False}, \text{False}]
              
        * Part of a single episode with terminal state in final transition:
        
          .. math::
              s_{1, 1}, s_{1, 2}, s_{1, 3}, s_{1, 4}, s_{1, 5}, s_{1, T}
              
          The corresponding ``done`` in the transitions should be:
          
          .. math::
              [\text{False}, \text{False}, \text{False}, \text{False}, \text{True}]
              
        * Two episodes (first episode terminates but second):
        
          .. math::
              s_{1, 1}, s_{1, 2}, s_{1, 3}, s_{1, T}, s_{2, 1}, s_{2, 2}, s_{2, 3}
              
          The corresponding ``done`` in the transitions should be:
          
          .. math::
              [\text{False}, \text{False}, \text{True}, \text{False}, \text{False}]
        
        * Two episodes (both episodes terminate):
        
          .. math::
              s_{1, 1}, s_{1, 2}, s_{1, 3}, s_{1, T}, s_{2, 1}, s_{2, 2}, s_{2, T}
        
          The corresponding ``done`` in the transitions should be:
          
          .. math::
              [\text{False}, \text{False}, \text{True}, \text{False}, \text{True}]
        
        Note that we do not restrict the first state to be initial state from an episode, this allows us to
        store the rolling episodic transitions into several segments. 
        
        For history containing transitions from a single episode, it is recommended to use :class:`Trajectory` instead.
    
    Example::
    
        >>> transition1 = Transition(s=10, a=-1, r=1, s_next=20, done=False)
        >>> transition1.add_info('V_s', torch.tensor(100.))

        >>> transition2 = Transition(s=20, a=-2, r=2, s_next=30, done=True)
        >>> transition2.add_info('V_s', torch.tensor(200.))
        >>> transition2.add_info('V_s_next', torch.tensor(250.))

        >>> transition3 = Transition(s=35, a=-3, r=3, s_next=40, done=False)
        >>> transition3.add_info('V_s', torch.tensor(300.))

        >>> transition4 = Transition(s=40, a=-4, r=4, s_next=50, done=False)
        >>> transition4.add_info('V_s', torch.tensor(400.))
        >>> transition4.add_info('V_s_next', torch.tensor(500.))

        >>> segment = Segment(gamma=0.1)
        >>> segment.add_transition(transition1)
        >>> segment.add_transition(transition2)
        >>> segment.add_transition(transition3)
        >>> segment.add_transition(transition4)

        >>> segment
        Segment: 
            Transition: (s=10, a=-1, r=1.0, s_next=20, done=False)
            Transition: (s=20, a=-2, r=2.0, s_next=30, done=True)
            Transition: (s=35, a=-3, r=3.0, s_next=40, done=False)
            Transition: (s=40, a=-4, r=4.0, s_next=50, done=False)
        
        >>> segment.all_s
        [10, 20, 30, 35, 40, 50]
        
        >>> segment.all_r
        [1.0, 2.0, 3.0, 4.0]
        
        >>> segment.all_done
        [False, True, False, False]
        
        >>> segment.all_V
        [tensor(100.),
         tensor(200.),
         tensor(250.),
         tensor(300.),
         tensor(400.),
         tensor(500.)]
         
        >>> segment.all_returns
        [3.0, 2.0, 7.0, 4.0]
        
        >>> segment.all_discounted_returns
        [3.0, 2.0, 507.0, 504.0]
        
        >>> segment.all_bootstrapped_returns
        [3.0, 2.0, 507.0, 504.0]
        
        >>> segment.all_bootstrapped_discounted_returns
        [1.2, 2.0, 8.4, 54.0]
        
        >>> segment.all_TD
        [-79.0, -198.0, -257.0, -346.0]
        
    """
    def __init__(self, gamma):
        r"""Initialize the segment of transitions. 
        
        .. note::
        
            Internally, it maintains a list of :class:`Trajectory` objects for each sub-segment of 
            episodic transitions. Each call of :meth:`add_transition` will add the transition to the 
            last Trajectory, when an episode terminates, a new :class:`Trajectory` will be created 
            and appended to the trajectory list. 
            The :meth:`transitions` will iterate over all transitions for each Trajectory object. 
            This might degrade the runtime speed but much easier to maintain the code and to add
            new functionality simply in :class:`Trajectory`. This will also reduce the risk of bugs
            because often the computations with segments alone are quite complicated. 
        
        Args:
            gamme (float): discount factor. 
        """
        self.gamma = gamma
        self.info = {}
        
        # Create trajectory buffer
        self.trajectories = [Trajectory(gamma=self.gamma)]
        
    @property
    def transitions(self):
        # Replace common property to allow calling self.transitions
        
        #########
        # Use itertools.chain().from_iterable, on average 50% faster than following method
        #transitions = []
        #for trajectory in self.trajectories:
        #    transitions.extend(trajectory.transitions)
        ########
        
        transitions = list(chain.from_iterable([trajectory.transitions for trajectory in self.trajectories]))
            
        return transitions
    
    def add_transition(self, transition):
        # If last transition terminates the last trajectory, then create and append a new Trajectory object
        if self.trajectories[-1].T > 0 and self.trajectories[-1].transitions[-1].done:
            self.trajectories.append(Trajectory(gamma=self.gamma))
        # Override the method, add transition to the last sub-trajectory
        self.trajectories[-1].add_transition(transition)
        
    @property
    def all_s(self):
        r"""Return a list of all states in the segment, including all intermediate terminal states
        (i.e. ``.s_next`` in each transition with ``done=True`` and last transition)
        """
        
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        all_s = list(chain.from_iterable([trajectory.all_s for trajectory in self.trajectories]))
        
        return all_s
    
    @property
    def all_returns(self):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = list(chain.from_iterable([trajectory.all_returns for trajectory in self.trajectories]))
        
        return out
    @property
    def all_discounted_returns(self):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = [trajectory.all_discounted_returns for trajectory in self.trajectories]
        out = list(chain.from_iterable(out))
        
        return out
    
    @property
    def all_bootstrapped_returns(self):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = [trajectory.all_bootstrapped_returns for trajectory in self.trajectories]
        out = list(chain.from_iterable(out))
        
        return out
    
    @property
    def all_bootstrapped_discounted_returns(self):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = [trajectory.all_bootstrapped_discounted_returns for trajectory in self.trajectories]
        out = list(chain.from_iterable(out))
        
        return out
    
    @property
    def all_V(self):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = [trajectory.all_V for trajectory in self.trajectories]
        out = list(chain.from_iterable(out))
        
        return out
    
    @property
    def all_TD(self):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = [trajectory.all_TD for trajectory in self.trajectories]
        out = list(chain.from_iterable(out))
        
        return out

    def all_GAE(self, gae_lambda):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = [trajectory.all_GAE(gae_lambda) for trajectory in self.trajectories]
        out = list(chain.from_iterable(out))
        
        return out
