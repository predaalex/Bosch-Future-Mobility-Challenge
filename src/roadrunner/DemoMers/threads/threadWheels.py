import threading
from src.templates.threadwithstop import ThreadWithStop
from src.utils.automata.driveDFA import DriveDFA
from src.utils.messages.allMessages import Brake, EngineRun, LaneDetection, SignDetection, SpeedMotor, SteerMotor
from src.utils.pid import pid_in, pid_out
from src.utils.messages.commands import sendMessageToQueue, subscribeToConfig
from src.utils.automata.signsDFA import SignsDFA

class threadWheels(ThreadWithStop):
    def __init__(self, queuesList, logging, pipeLaneRecv, pipeLaneSend, pipeSignRecv, pipeSignSend):
        super(threadWheels, self).__init__()

        self.queuesList = queuesList
        self.logging = logging
        
        self.pipeLaneRecv = pipeLaneRecv
        self.pipeLaneSend = pipeLaneSend

        self.pipeSignRecv = pipeSignRecv
        self.pipeSignSend = pipeSignSend

        # subscribe to LaneDetection messages
        self.subscribe()

        self.engineRunning = False

        # all the DFAs will be handled here
        self.driveDFA = DriveDFA()
        self.lastDriveState = self.driveDFA.current_state

        self.signDFA = SignsDFA()
        self.lastSignState = self.signDFA.current_state
        self.signDFA.pump_state(self.signDFA.STOP, 2,{self.signDFA.fail: self.signDFA.WATCH})
        self.signDFA.pump_state(self.signDFA.PARKING, 2,{self.signDFA.fail: self.signDFA.WATCH})
        self.signDFA.pump_state(self.signDFA.PRIORITY, 2,{self.signDFA.fail: self.signDFA.WATCH})
        self.signDFA.pump_state(self.signDFA.CROSSWALK, 2,{self.signDFA.fail: self.signDFA.WATCH})
        
        #after qualifications
        self.signDFA.pump_state(self.signDFA.HIGHWAY_ENTRANCE, 3,{self.signDFA.fail: self.signDFA.WATCH})
        self.signDFA.pump_state(self.signDFA.HIGHWAY_EXIT, 3,{self.signDFA.fail: self.signDFA.WATCH})
        self.signDFA.pump_state(self.signDFA.ROUND_ABOUT, 3,{self.signDFA.fail: self.signDFA.WATCH})
        self.signDFA.pump_state(self.signDFA.ONE_WAY, 3,{self.signDFA.fail: self.signDFA.WATCH})
        self.signDFA.pump_state(self.signDFA.NO_ENTRY, 3,{self.signDFA.fail: self.signDFA.WATCH})
        # TODO add the pumping for the signsDFA

        # TODO IMPORTANT
        # stop 3 secunde, crosswalk, priority, parking

        #TODO semaphore DFA

        #TODO intersection DFA

        #TODO parking DFA

        self.lastSteer = 0

        # flag to signal to Automata Apply to call or not a timer
        self.waitingForTimer = False
    

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway. In this case, we subscribe to LaneDetection messages."""
        subscribeToConfig(self.queuesList, LaneDetection, "threadWheels", self.pipeLaneSend)

        subscribeToConfig(self.queuesList, SignDetection, "threadWheels", self.pipeSignSend)


    def delayed_execution(self, delay_seconds, function_to_execute, *args):
        """
        Executes a given function after a specified delay in seconds.

        Parameters:
        - delay_seconds (int): The number of seconds to wait before executing the function.
        - function_to_execute (function): The function to execute after the delay.
        - *args: Arguments to pass to the function_to_execute.
        """
        # Create a Timer thread that will execute the function after the delay
        timer_thread = threading.Timer(delay_seconds, function_to_execute, args=args)
        timer_thread.start() 

    def go_car_normal(self):
        self.driveDFA(DriveDFA.normal)
        self.waitingForTimer = False
        # reset the state
        self.signDFA(SignsDFA.fail)

    def reset_state(self):
        self.driveDFA(DriveDFA.normal)
        self.waitingForTimer = False
        # reset the state
        self.signDFA(SignsDFA.fail)

    def run(self):
        while self._running:

            ###################### TEST COMMANDS ######################
            # self.exampleWithoutDFA()
            # self.exampleDFA()

            ###################### DIRECTION CONTROL ######################
            if self.pipeLaneRecv.poll():
                msg = self.pipeLaneRecv.recv()
                print(f"Received lane message: {msg}")

                action = msg['value']['action']
                if action == "laneSteer":
                    if not self.engineRunning:
                        self.start_engine()
                        self.engineRunning = True

                    self.steer(msg['value']['steer'])
            
            ###################### SIGNS CONTROL ######################
            if self.pipeSignRecv.poll():
                msg = self.pipeSignRecv.recv()
                print(f"Received sign message: {msg}")

                action = msg['value']['action']
                if action == "signDetect":
                    if not self.engineRunning:
                        self.start_engine()
                        self.engineRunning = True


                    # we aslo should check the accuracy and the area of the rectangle
                    # to calculate the distance

                    # if the current state is WATCH, we need to check if the sign is STOP, PARKING, PRIORITY or CROSSWALK
                    if self.signDFA.get_current_state(False) == SignsDFA.WATCH:
                        if msg['value']['sign_type'] == SignsDFA.STOP:
                            self.signDFA(SignsDFA.see_stop)
                        if msg['value']['sign_type'] == SignsDFA.PARKING:
                            self.signDFA(SignsDFA.see_parking)
                        if msg['value']['sign_type'] == SignsDFA.PRIORITY:
                            self.signDFA(SignsDFA.see_priority)
                        if msg['value']['sign_type'] == SignsDFA.CROSSWALK:
                            self.signDFA(SignsDFA.see_crosswalk)
                    else:
                        # if the current state is not WATCH, we need to check if the sign is the same as the current state
                        if self.signDFA.get_current_state(False) != msg['value']['sign_type']:
                            # if the sign is not the same as the current state, we need to go to the WATCH state
                            if self.signDFA.is_pumped():
                                self.signDFA(SignsDFA.fail)
                            else:
                                # it saw another sign while being in a current validated sign state.
                                # TODO handle this
                                pass
                        else:
                            self.signDFA(SignsDFA.tick)

                print(f"Current sign state: {self.signDFA.get_current_state()}")

            ###################### SIGN AUTOMATA APPLY ######################
            if not self.waitingForTimer:
                if self.signDFA.get_current_state(real=True) == SignsDFA.STOP:
                    self.driveDFA(DriveDFA.STOP)
                    self.waitingForTimer = True
                    #after 3 seconds start the car
                    self.delayed_execution(3, self.reset_state)
                elif self.signDFA.get_current_state(real=True) == SignsDFA.PARKING:
                    pass
                    # TODO call timer for parking 
                    # self.park_car()
                elif self.signDFA.get_current_state(real=True) == SignsDFA.PRIORITY:
                    self.driveDFA(DriveDFA.DRIVE_NORMAL)
                    self.waitingForTimer = True
                    # after 3 seconds reset the states
                    # NEEDED TO RESET waitingForTimer AND signDFA CURRENT STATE
                    # !!!! DO NOT REMOVE !!!
                    self.delayed_execution(3, self.reset_state)
                elif self.signDFA.get_current_state(real=True) == SignsDFA.CROSSWALK:
                    self.driveDFA(DriveDFA.DRIVE_SLOW)
                    self.waitingForTimer = True
                    # after 5 seconds reset the states
                    self.delayed_execution(5, self.reset_state)
            

            ###################### WHEELS CONTROL ######################
            if self.lastDriveState != self.driveDFA.get_current_state():
                self.lastDriveState = self.driveDFA.get_current_state()

                if self.driveDFA.get_current_state() == DriveDFA.STOP:
                    self.brake()
                
                if self.driveDFA.get_current_state() == DriveDFA.DRIVE_SLOW:
                    # self.speed(1)
                    # self.brake()
                    self.speed(10)

                if self.driveDFA.get_current_state() == DriveDFA.DRIVE_NORMAL:
                    self.speed(15)
                    # self.brake()
                
                if self.driveDFA.get_current_state() == DriveDFA.DRIVE_FAST:
                    self.speed(20)

                if self.driveDFA.get_current_state() == DriveDFA.REVERSE:
                    self.speed(-10)

                

                print(f"Current state: {self.driveDFA.get_current_state()}")


    def exampleDFA(self):
        if not self.engineRunning:
            self.start_engine()
            self.engineRunning = True
        self.driveDFA(self.driveDFA.normal)
        self.steer(20)
                    
    def exampleWithoutDFA(self):
        if not self.engineRunning:
            sendMessageToQueue(self.queuesList, EngineRun, True)
            self.engineRunning = True
        self.speed(10)
        self.steer(20)

    def start_engine(self):
        sendMessageToQueue(self.queuesList, EngineRun, True)

        if self.driveDFA.get_current_state() == self.driveDFA.STOP:
            self.driveDFA(self.driveDFA.start_engine)

    def speed(self, speed):
        sendMessageToQueue(self.queuesList, SpeedMotor, speed)

    def brake(self):
        sendMessageToQueue(self.queuesList, Brake, 0)

    def steer(self, steer):
        print("steer value", steer)
        sendMessageToQueue(self.queuesList, SteerMotor, steer)


    def stop(self):
        super(threadWheels, self).stop()