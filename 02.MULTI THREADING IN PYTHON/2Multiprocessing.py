import multiprocessing
import time

def squre_num():
    for i in range(5):
        print(f"Square: {i*i}")
        time.sleep(1)

def cubic_num():
    for i in range(5):
        print(f"Cube: {i*i*i}")
        time.sleep(1)

if __name__ == "__main__":
    ## create 2 process
    p1 = multiprocessing.Process(target=squre_num)
    p2 = multiprocessing.Process(target=cubic_num)

    t = time.time()   # ✅ function call

    ## start the process
    p1.start()
    p2.start()

    ## wait for the process to complete
    p1.join()
    p2.join()

    finish_time = time.time() - t   # ✅ correct
    print("Execution Time:", finish_time)