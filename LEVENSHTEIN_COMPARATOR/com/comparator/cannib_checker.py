'''
Created on 15 Jul 2015

@author: sduprey
'''
import Levenshtein
from my_levenshtein_thread_computer import my_levenshtein_thread_computer

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


lf = open("/home/sduprey/My_Data/My_Cannib_Checker/lf.txt", "r")
sdx = open("/home/sduprey/My_Data/My_Cannib_Checker/sdx.txt", "r")

# you can't write parallely into the same file : database
#out = open("/home/sduprey/My_Data/My_Cannib_Checker/out.txt", "w")

parself = lf.readlines()
parsesdx = sdx.readlines()

count = sum(1 for line in parself)


print("Length of the firs list to loop open :")
print(str(len(parself)))
print(str(count))


chunk_size = 1500
my_chunks = list(chunks(parself, chunk_size))
nb_pieces_chunk = len(my_chunks)
print('Delegating to '+str(nb_pieces_chunk) + ' threads ')
   
threads = []
thread_counter = 0 
for i in range(nb_pieces_chunk) :      
    thread_counter = thread_counter + 1
    print('Launching a thread number ' + str(thread_counter) + ' for the categories : ' + str(my_chunks[i]))
    # Create new threads
    threadi = my_levenshtein_thread_computer(i, "Thread_"+str(i), my_chunks[i], parsesdx)
    threadi.start()
    threads.append(threadi)
    
    # Wait for all threads to complete
for t in threads:
    t.join()


print "Exiting Main Thread after all threads finish"

lf.close()
sdx.close()
# go for a database
#out.close()