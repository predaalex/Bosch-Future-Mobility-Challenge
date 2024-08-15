from src.utils.automata.DFAInterface import DFAInterface
#from DFAInterface import DFAInterface
class SignsDFA(DFAInterface):
    # States
    WATCH = "watch"
    STOP = "stop"
    PARKING = "parking"
    PRIORITY = "priority"
    CROSSWALK = "cross_walk"
    HIGHWAY_ENTRANCE = "highway_entrance"
    HIGHWAY_EXIT = "highway_exit"
    ROUND_ABOUT = "round_about"
    ONE_WAY = "one_way"
    NO_ENTRY = "no_entry"
    
    # Actions
    fail = "fail"
    see_stop = "see_stop"
    see_parking = "see_parking"
    see_priority = "see_priority"
    see_crosswalk = "see_crosswalk"
    see_highway_entrance = "see_highway_entrance"
    see_highway_exit = "see_highway_exit"
    see_round_about = "see_round_about"
    see_one_way = "see_one_way"
    see_no_entry = "see_no_entry"

    transitions = {
        WATCH: {see_stop: STOP,
                see_parking: PARKING,
                see_priority: PRIORITY,
                see_crosswalk: CROSSWALK,
                see_highway_entrance: HIGHWAY_ENTRANCE,
                see_highway_exit: HIGHWAY_EXIT,
                see_round_about: ROUND_ABOUT,
                see_one_way: ONE_WAY,
                see_no_entry: NO_ENTRY},
        STOP: {fail: WATCH},
        PARKING: {fail: WATCH},
        PRIORITY: {fail: WATCH},
        CROSSWALK: {fail: WATCH},
        HIGHWAY_ENTRANCE: {fail: WATCH},
        HIGHWAY_EXIT: {fail: WATCH},
        ROUND_ABOUT: {fail: WATCH},
        ONE_WAY: {fail: WATCH},
        NO_ENTRY: {fail: WATCH}
    }

    def __init__(self):
        super().__init__(SignsDFA.transitions, SignsDFA.WATCH)
        
def __main__():
    # Test
    signs_dfa = SignsDFA()

    ######################

    # signs_dfa.get_current_state(real=False) == STOP

    #####################
    
    #This does not work...
    # for sign in signs_dfa.transitions:
    #     if sign == signs_dfa.WATCH:
    #         continue
    #     signs_dfa.pump_state(sign,signs_dfa.inter_no,{signs_dfa.fail: signs_dfa.WATCH})

    signs_dfa.pump_state(signs_dfa.STOP, 3,{signs_dfa.fail: signs_dfa.WATCH})
    print(signs_dfa.transitions)
    print(signs_dfa(signs_dfa.see_stop))
    print(signs_dfa(signs_dfa.tick))
    print(signs_dfa(signs_dfa.tick))
    print(signs_dfa(signs_dfa.fail))
    print(signs_dfa(signs_dfa.tick))
    print(signs_dfa(signs_dfa.fail))
    # print(signs_dfa.transition(signs_dfa.speed_up))
    # print(signs_dfa.transition(signs_dfa.slow_down,))
    # print(signs_dfa.transition(signs_dfa.emergency_stop))

# __main__()