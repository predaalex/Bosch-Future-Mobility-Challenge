from multiprocessing import Pipe
from src.roadrunner.DemoMers.threads.threadWheels import threadWheels
from src.templates.workerprocess import WorkerProcess


class processDemoMers(WorkerProcess):
    # ====================================== INIT ==========================================
    def __init__(self, queueList, logging):
        self.queuesList = queueList
        self.logging = logging

        pipeLaneRecv, pipeLaneSend = Pipe(duplex=False)
        self.pipeLaneRecv = pipeLaneRecv
        self.pipeLaneSend = pipeLaneSend

        pipeSignRecv, pipeSignSend = Pipe(duplex=False)
        self.pipeSignRecv = pipeSignRecv
        self.pipeSignSend = pipeSignSend

        super(processDemoMers, self).__init__(self.queuesList)

    # ===================================== STOP ==========================================
    def _stop(self):
        for thread in self.threads:
            thread.stop()
            thread.join()
        super(processDemoMers, self).stop()

    # ===================================== RUN ==========================================
    def run(self):
        super(processDemoMers, self).run()

    # ===================================== INIT TH ======================================
    def _init_threads(self):

        WhTh = threadWheels(
            self.queuesList, self.logging, self.pipeLaneRecv, self.pipeLaneSend, self.pipeSignRecv, self.pipeSignSend
        )

        self.threads.append(WhTh)
