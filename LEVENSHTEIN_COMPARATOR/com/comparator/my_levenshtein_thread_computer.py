'''
Created on 17 Jul 2015

@author: sduprey
'''

import threading
import Levenshtein

class my_levenshtein_thread_computer (threading.Thread):
    def __init__(self, threadID, name, my_list_chunk, my_second_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.my_list_chunk = my_list_chunk
        self.my_second_list =my_second_list
    
    def run(self):
        print "Starting " + self.name
        stepping_through_list(self.name, self.my_list_chunk, self.my_second_list)
        print "Exiting " + self.name

def stepping_through_list(threadName, my_list_chunk, my_second_list):
    print "%s beginning to deal with its categories %s" % (threadName, str(my_list_chunk))

    print(threadName + 'Testing sample size')


    print(threadName +' Training and predicting xgb classifier for each category')
    
    
    i = 0
    for kwlf in my_list_chunk:
        print(threadName + " : Dealing with entry :"+str(i))
        print(threadName + " : Dealing with entry :"+kwlf)
        for kwsdx in my_second_list:
            kwsdx = kwsdx.strip()
            kwlf = kwlf.strip()
            ratio = Levenshtein.ratio(kwlf, kwsdx)
            if ratio > 0.9:
                  #out.write()
                 print(kwlf+"|"+kwsdx+"|"+str(ratio)+"\n")
                   #print(kwlf+" "+kwsdx+" "+str(ratio))
            #print(str(i)+" "+str((i/count) * 100)+" %")
        i += 1
    
  #        counter -= 1