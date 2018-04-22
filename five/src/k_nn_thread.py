import queue, time, threading


class KNN_Thread(threading.Thread):
    def __init__(self, threadID, name, q, processingMethod, lock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

        self.exitFlag = 0
        self.queueLock = lock
        self.processingMethod = processingMethod


    def run(self):
        print("Starting " + self.name)
        # Do the processing of data
        while not self.exitFlag:
            self.queueLock.acquire()
            if not self.q.empty():
                case, label = self.q.get()
                self.queueLock.release()
                print("{} processing...".format(self.name))
                self.processingMethod(case, label)

            else:
                self.queueLock.release()
                time.sleep(1)


        print("Exiting " + self.name)


    def setExitFlag(self, value):
        self.exitFlag = value


