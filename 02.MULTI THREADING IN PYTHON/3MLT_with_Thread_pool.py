## Multithreading with thread pool Executor

from concurrent.futures import ThreadPoolExecutor
import time

def print_number(number):
    time.sleep(1)
    return f"number:{number}"

number = [1,2,3,4,5,6,6,7]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(print_number,number)

for result in results:
    print(result)