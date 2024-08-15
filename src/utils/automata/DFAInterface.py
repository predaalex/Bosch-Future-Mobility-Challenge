# 
class DFAInterface:
    """
    Represents a DFA (Deterministic Finite Automata) interface.

    Attributes:
        transitions (dict): A dictionary representing the transition table of the DFA.
        current_state (str): The current state of the DFA.

    Methods:
        transition(input_signal): Transitions to the next state based on the input signal.
        pump_state(state, n, important={}): Pumps the specified state by creating additional states.
        get_current_state(real=True): Returns the current state of the DFA.
        is_pumped(): Checks if the current state is a pumped state.
    """
    tick = "tick"

    def __init__(self, transitions, current_state):
        self.transitions = transitions
        self.current_state = current_state
    
    def transition(self, input_signal):
        """
        Transition to the next state based on the input signal.

        Args:
            input_signal (str): The input signal to be processed.

        Returns:
            str or None: The next state if a valid transition exists, None otherwise.
        """
        next_state = self.transitions.get(self.current_state, {}).get(input_signal)
        if next_state:
            self.current_state = next_state
            return self.current_state
        else:
            return None
        
    def pump_state(self, state, n, important={}):
        """
        Pumps the specified state by creating additional states.

        Args:
            state (str): The state to pump.
            n (int): The number of additional states to create.
            important (dict, optional): Additional transitions to add to the pumped states. Defaults to {}.
        """
        for k in self.transitions:
            for k2 in self.transitions[k]:
                if self.transitions[k][k2] == state:
                    self.transitions[k][k2] = "P-"+state+"-0"
        for i in range(n):
            transition_dict = {self.tick: state} if i == n-1 else {self.tick: "P-"+state+"-"+str(i+1)}
            transition_dict.update(important)
            self.transitions["P-"+state+"-"+str(i)] = transition_dict

    def get_current_state(self, real=True):
        """
        Returns the current state of the DFA.

        Args:
            real (bool, optional): Specifies whether to return the real state or the pumped state. Defaults to True.

        Returns:
            str: The current state of the DFA.
        """
        if real:
            return self.current_state
        elif self.is_pumped():
            return self.current_state.split("-")[1]
        else:
            return self.current_state
    
    def is_pumped(self):
        """
        Checks if the current state is a pumped state.

        Returns:
            bool: True if the current state is a pumped state, False otherwise.
        """
        return self.current_state.startswith("P-")

    def __call__(self, value):
        return self.transition(value)