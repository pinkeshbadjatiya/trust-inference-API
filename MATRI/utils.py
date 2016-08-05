from __future__ import print_function
import sys

class log:
    """ Class to log messages to terminal """

    def __init__(self):
        self.head = None
        self.msg = None
        self.__buff = " "*50

    def updateHEAD(self, s):
        self.nextLine()
        self.head = s
        self.msg = ""
        self.display()

    def updateMSG(self, s):
        self.msg = s
        self.display()

    def display(self):
        if self.head is None:
            self.head = ""
        if self.msg is None:
            self.msg = ""
        print("\r%s: %s %s" % (self.head, self.msg, self.__buff), end="")
        sys.stdout.flush()

    def nextLine(self):
        print("")
        sys.stdout.flush()
