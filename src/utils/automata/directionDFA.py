from src.utils.automata.DFAInterface import DFAInterface
#from DFAInterface import DFAInterface
class DirectionDFA(DFAInterface):
    # States
    NEUTRAL = "neutral"
    APPLY = "apply"
    
    # Actions
    fail = "fail"
    steer = "steer"
    reset = "reset"

    transitions = {
        NEUTRAL:{steer:APPLY},
        APPLY:{reset:NEUTRAL}
    }

    def __init__(self):
        super().__init__(DirectionDFA.transitions, DirectionDFA.NEUTRAL)
        
if __name__ == "__main__":
    # Test
    direction_dfa = DirectionDFA()

    ######################

    # direction_dfa.get_current_state(real=False) == STOP

    #####################

    direction_dfa.pump_state(direction_dfa.APPLY, 3,{direction_dfa.fail: direction_dfa.NEUTRAL})
    print(direction_dfa.transitions)
    print(direction_dfa(direction_dfa.steer))
    print(direction_dfa(direction_dfa.tick))
    print(direction_dfa(direction_dfa.tick))
    print(direction_dfa(direction_dfa.tick))
    print(direction_dfa(direction_dfa.reset))
    # print(direction_dfa.transition(direction_dfa.fail))
    # print(direction_dfa.transition(direction_dfa.speed_up))
    # print(direction_dfa.transition(direction_dfa.slow_down,))
    # print(direction_dfa.transition(direction_dfa.emergency_stop))

# __main__()