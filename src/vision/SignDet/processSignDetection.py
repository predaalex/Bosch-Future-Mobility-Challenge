from multiprocessing import Pipe
from src.templates.workerprocess import WorkerProcess
from src.vision.SignDet.threads.threadSignDetection import threadSignDetection


class processSignDetection(WorkerProcess):
    # ====================================== INIT ==========================================
    def __init__(self, queueList, logging):
        self.queuesList = queueList
        self.logging = logging

        pipeSignRecv, pipeSignSend = Pipe(duplex=False)
        self.pipeSignRecv = pipeSignRecv
        self.pipeSignSend = pipeSignSend
        super(processSignDetection, self).__init__(self.queuesList)

    # ===================================== STOP ==========================================
    def _stop(self):
        for thread in self.threads:
            thread.stop()
            thread.join()
        super(processSignDetection, self).stop()

    # ===================================== RUN ==========================================
    def run(self):
        super(processSignDetection, self).run()

    # ===================================== INIT TH ======================================
    def _init_threads(self):
        SdTh = threadSignDetection(
            self.queuesList, self.logging, self.pipeSignRecv, self.pipeSignSend
        )

        self.threads.append(SdTh)
