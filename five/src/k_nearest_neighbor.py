"""k_nearest_neighbor.py: ML algorithm for predicting class of handwritten character."""

__author__      = "Sigve Skaugvoll"
__copyright__   = "Copyright 2018, skaugvoll.com"
__license__     = "MIT"
__version__     = "1.0.0"
__maintainer__  = "Sigve Skaugvoll"
__email__       = "sigve@skaugvoll.com"
__status__      = "Development"

from queue import PriorityQueue, Queue
from scipy.spatial.distance import euclidean
import operator, time, sys, threading
from itertools import count

from data_generator import DataGenerator
from k_nn_thread import KNN_Thread




class KNN():
    def __init__(self, training_cases, training_labels, testing_cases, testing_labels, k=10):
        self.training_cases = training_cases
        self.training_labels = training_labels
        self.testing_cases = testing_cases
        self.testing_labels = testing_labels

        self.nearest_neighbors= None # used to keep track of which neighbors are closest
        self.tiebreaker = None # this if for fixing some numpy shit, for priorityQueue, the entries before case
                               # cannot be eequal, because then it tries to match the cases, which uses a truth-table
                               # which causes an heap error. thus this is to make unique attribute so that each entry
                               # can be compared.

        self.k = k

        self.correct = 0
        self.wrong = 0


    # Step 1: find neighbors
    def find_neighbors(self, caseToClassify):
        for case, label in zip(self.training_cases, self.training_labels):
            self.nearest_neighbors.put((euclidean(caseToClassify, case), next(self.tiebreaker), case,  label))


    # Step 2: find classification based on k neighbors
    def classification(self):
        classes = {}
        for neighbor in range(0, self.k):
            nn = self.nearest_neighbors.get() # returns (distance, neighbor, label)
            # print("NN: {}".format(nn))
            nn_class = nn[-1]

            if nn_class in classes:
                classes[nn_class] += 1
            else:
                classes[nn_class] = 1


        prediction = max(classes.items(), key=operator.itemgetter(1))

        prediction_value = prediction[0]
        prediction_num_neighbors = prediction[1]
        return prediction_value, prediction_num_neighbors


    # Step 3: Run this badboy
    def run(self):
        i = 1
        l = len(self.testing_cases)
        start = time.time()
        for case, label in zip(self.testing_cases, self.testing_labels):
            # print("Test case #{} / {} -- {:.2f}%".format(i, l, float(i)/float(l)))

            self.nearest_neighbors = PriorityQueue()
            self.tiebreaker = count()
            self.find_neighbors(case)
            pred, vote = self.classification()

            if pred == label:
                self.correct += 1
            else:
                self.wrong += 1

            i += 1

        end = time.time()
        print("\nTime used: {:.2f}m\nAccuracy: {:.2f}\nCorrect: {} \nWrong: {}".format((end - start)/60, (self.correct / len(self.testing_cases)) * 100, self.correct, self.wrong))


    def threaded_processing(self, caseToClassify, caseToClassify_label):
        thread_mem = threading.local()
        thread_mem.nearest_neighbors = PriorityQueue()
        thread_mem.tiebreaker = count()

        # find neighbors
        for case, label in zip(self.training_cases, self.training_labels):
            thread_mem.nearest_neighbors.put((euclidean(caseToClassify, case), next(thread_mem.tiebreaker), case, label))

        # classification
        thread_mem.classes = {}
        for neighbor in range(0, self.k):
            nn = thread_mem.nearest_neighbors.get()  # returns (distance, neighbor, label)
            nn_class = nn[-1]

            if nn_class in thread_mem.classes :
                thread_mem.classes[nn_class] += 1
            else:
                thread_mem.classes[nn_class] = 1

        thread_mem.prediction = max(thread_mem.classes .items(), key=operator.itemgetter(1))[0]

        if thread_mem.prediction == caseToClassify_label:
            self.correct += 1
        else :
            self.wrong += 1



    def run_threaded(self, num_threads=8):
        queueLock = threading.Lock()
        workQueue = Queue(len(self.testing_cases))
        threads = []

        # Create new threads
        for id in range(1, num_threads+1):
            thread = KNN_Thread(id, "Thread-{}".format(id), workQueue, self.threaded_processing, queueLock)
            thread.start()
            threads.append(thread)

        # Fill the queue
        queueLock.acquire()
        for case, label in zip(self.testing_cases, self.testing_labels):
            task = (case, label)
            workQueue.put(task)
        queueLock.release()

        start = time.time()
        # Wait for queue to empty
        while not workQueue.empty():
            # print("Remaining: {}".format(workQueue.qsize()))
            pass

        # Notify threads it's time to exit
        for t in threads:
            t.setExitFlag(1)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        end = time.time()

        print("\nTime used: {:.2f}m\nAccuracy: {:.2f}\nCorrect: {} \nWrong: {}".format((end - start)/60, (self.correct / len(self.testing_cases)) * 100, self.correct, self.wrong))




    def change_test_data(self, test_cases, test_labels):
        self.testing_cases = test_cases
        self.testing_labels = test_labels

    def change_k(self, k):
        if k < 1:
            print("K must be greater than 1")
            sys.exit(0)
        self.k = k

    def reset(self):
        self.nearest_neighbors = None  # used to keep track of which neighbors are closest
        self.tiebreaker = None
        self.correct = 0
        self.wrong = 0



def main():
    dg = DataGenerator()
    dg.shuffle_data()

    training_cases, training_labels = dg.get_training_partition(percentage=.8)
    testing_cases, testing_labels = dg.get_testing_partition(percentage=.2)

    clf = KNN(training_cases, training_labels, testing_cases, testing_labels, k=10)

    clf.run()
    print("\nNOW Threaded")
    clf.run_threaded(8)






if __name__ == "__main__":
    main()