from simple_pid import PID

# declare the pid
pid = PID(1, 0, 0, setpoint=0) #arbitrary set Kp Ki Kd, TODO choose better values
#the value that will be returned by pid
control = 0 

#input the error into the pid
def pid_in(error):
    global control
    control = pid(error)

#returns the value from the pid
def pid_out():
    return control
