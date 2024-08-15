from multiprocessing import Pipe

from src.templates.workerprocess import WorkerProcess
from src.vision.LaneDet.threads.threadLaneDetection import threadLaneDetection


class processLaneDetection(WorkerProcess):
    # ====================================== INIT ==========================================
    def __init__(self, queueList, logging):
        self.queuesList = queueList
        self.logging = logging

        pipeLaneRecv, pipeLaneSend = Pipe(duplex=False)
        self.pipeLaneRecv = pipeLaneRecv
        self.pipeLaneSend = pipeLaneSend

        super(processLaneDetection, self).__init__(self.queuesList)

    # ===================================== STOP ==========================================
    def _stop(self):
        for thread in self.threads:
            thread.stop()
            thread.join()
        super(processLaneDetection, self).stop()

    # ===================================== RUN ==========================================
    def run(self):
        super(processLaneDetection, self).run()

    # ===================================== INIT TH ======================================
    def _init_threads(self):
        LdTh = threadLaneDetection(
            self.queuesList, self.logging, self.pipeLaneRecv, self.pipeLaneSend
        )
        self.threads.append(LdTh)

