import time

def time_since(start_time):
    return time.time()-start_time

if __name__ == '__main__':
    start_time = time.time()
    time.sleep(5)
    print(time_since(start_time))
