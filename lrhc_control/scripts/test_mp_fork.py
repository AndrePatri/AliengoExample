import psutil
import time
import os
import multiprocessing as mp


def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / (1024**3)  # Convert bytes to gigabytes
    print(f"[PID {os.getpid()}] Memory usage now: {memory_usage_gb} GB")
    return memory_usage_gb
    
def child_process():
    """Child process imports NumPy and does nothing."""
    import numpy as np
    time.sleep(20)
    get_memory_usage()

def get_system_memory(label="", prev=0.0):
    """Get system memory usage."""
    mem = psutil.virtual_memory()
    used=mem.used / (1024 ** 3)
    print(f"{label} System Memory: Used = {used} GB, Available = {mem.available / (1024 ** 3)} GB,differece {used-prev}")
    return used


if __name__ == "__main__":
    n = 100  # Number of child processes
    
    class Prova():
        def __init__(self):
            a=1
        def import_aux(self):
            before=get_memory_usage()
            global np
            import numpy as np
            time.sleep(1)
            after=get_memory_usage()
            print(f"np weights {after-before}")

    pr=Prova()
    pr.import_aux()
    a=np.zeros((2, 1))
    time.sleep(1)
    system_start=get_system_memory(label="START")
    
    print("\nSpawning child processes...\n")

    mp.set_start_method("fork")  # Use the fork context
    processes = []
    for _ in range(n):
        p = mp.Process(target=child_process)
        processes.append(p)
        p.start()

    # Monitor child processes
    parent = psutil.Process(os.getpid())
    while all(p.is_alive() for p in processes):
        children = parent.children()
        print(f"\n[Parent PID {os.getpid()}] Monitoring child processes...")
        get_system_memory(label="END",prev=system_start)
        time.sleep(1)
    
    time.sleep(5)

    for p in processes:
        p.join()
    
    print("\nAll child processes have completed.")