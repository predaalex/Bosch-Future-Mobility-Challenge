
import base64
import time
import cv2

import numpy as np
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import SignDetection, mainCamera, serialCamera
from src.utils.messages.commands import subscribeToConfig, sendMessageToQueue
from src.utils.automata.signsDFA import SignsDFA


class threadSignDetection(ThreadWithStop):
    def __init__(self, queuesList, logging, pipeRecv, pipeSend):
        super(threadSignDetection, self).__init__()

        self.queuesList = queuesList
        self.logging = logging
        self.pipeRecv = pipeRecv
        self.pipeSend = pipeSend


    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway.
        In this case, we subscribe to Camera messages."""

        #important: camera sends 2 images, check the threadCamera, from line 165 for reference
        #choose wich camera to subscribe, you can remove one of the following subscriptions

        #main camera
        # (2048, 1080) pixels
        # subscribeToConfig(self.queuesList, mainCamera, "threadSignDetection", self.pipeSend)

        #serial camera
        # (480, 360) pixels
        subscribeToConfig(self.queuesList, serialCamera, "threadSignDetection", self.pipeSend)

    def run(self):
        # receive camera base64 image
        # check threadCamera, line 165 for reference
        # you receive 2 images, check the above line for reference

        #check if there is a message in the pipe
        while self._running:
            #testing
            # TEST COMMANDS
            # self.sendSignToWheels(SignsDFA.STOP, 1, 0, 0, 0, 0)
            # time.sleep(2)

            # self.sendSignToWheels(SignsDFA.CROSSWALK, 1, 0, 0, 0, 0)
            # time.sleep(2)

            # self.sendSignToWheels(SignsDFA.STOP, 1, 0, 0, 0, 0)
            # time.sleep(2)
            # self.sendSignToWheels(SignsDFA.STOP, 1, 0, 0, 0, 0)
            # time.sleep(2)
            # self.sendSignToWheels(SignsDFA.STOP, 1, 0, 0, 0, 0)
            # time.sleep(2)
            # self.sendSignToWheels(SignsDFA.STOP, 1, 0, 0, 0, 0)
            # time.sleep(2)

            # self.sendSignToWheels(SignsDFA.PARKING, 1, 0, 0, 0, 0)
            # time.sleep(2)

            if self.pipeRecv.poll():
                msg = self.pipeRecv.recv()
                # msg is a base64 image, again check line 165 from threadCamera for reference

                # process the image and send the object to wheels
                # object should include sign_type,x,y,width,height of the detected box
                
                frame_encoded = msg["value"]
                decoded_data = base64.b64decode(frame_encoded)
                np_data = np.fromstring(decoded_data,np.uint8)
                img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # now you can use the opencv image
    

    def sendSignToWheels(self, sign_type, confidence, x, y, width, height):
        sendMessageToQueue(self.queuesList, SignDetection,
                    {"action": "signDetect",
                         "sign_type": sign_type,
                         "confidence": confidence,
                         "x": x,
                         "y": y,
                         "width": width,
                         "height": height
                    })