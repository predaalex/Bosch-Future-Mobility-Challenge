from src.utils.automata.DFAInterface import DFAInterface


class DriveDFA(DFAInterface):


    STOP = "stop"
    DRIVE_FAST = "drive_fast"
    DRIVE_NORMAL = "drive_normal"
    DRIVE_SLOW = "drive_slow"
    REVERSE = "reverse"

    emergency_stop = "stop"
    start_engine = "start_engine"
    fast = "fast"
    normal = "normal"
    slow = "slow"
    back_up = "back_up"

    transitions = {
        STOP: {slow: DRIVE_SLOW, normal: DRIVE_NORMAL, fast: DRIVE_FAST, back_up: REVERSE, start_engine: DRIVE_NORMAL},
        DRIVE_SLOW: {emergency_stop: STOP, fast: DRIVE_FAST, normal: DRIVE_NORMAL},
        DRIVE_NORMAL: {emergency_stop: STOP, slow: DRIVE_SLOW, fast: DRIVE_FAST},
        DRIVE_FAST: {emergency_stop: STOP, slow: DRIVE_SLOW, normal: DRIVE_NORMAL},
        REVERSE: {emergency_stop: STOP}
    }

    def __init__(self):
        super().__init__(DriveDFA.transitions, DriveDFA.STOP)
        
    

def __main__():
    # Test
    drive_dfa = DriveDFA()

    ######################

    # drive_dfa.get_current_state(real=False) == STOP

    ######################

    drive_dfa.pump_state(drive_dfa.STOP, 5)
    drive_dfa.pump_state(drive_dfa.DRIVE_SLOW, 2, {drive_dfa.emergency_stop: drive_dfa.STOP})
    print(drive_dfa.transitions)
    print(drive_dfa(drive_dfa.tick))
    print(drive_dfa(drive_dfa.tick))
    print(drive_dfa(drive_dfa.emergency_stop))
