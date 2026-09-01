# Multithreading

import threading
import time

def print_numbers():
    for i in range(20):
        time.sleep(0.5)
        print(f"Number:{i}")

def print_letters():
    for letter in "abcdefghiklmnopqrstuv":
        time.sleep(0.4)
        print(f"Letters:{letter}")


# create 2 threds
t1 = threading.Thread(target=print_numbers)
t2 = threading.Thread(target=print_letters)

t= time.time()
t1.start()
t2.start()

# wait for the threads complete
t1.join()
t2.join()

finish_time = time.time()-t
print(finish_time)