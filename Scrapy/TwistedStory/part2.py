# ~*~ Twisted - A python tale ~*~
# Part 2
from time import sleep
import threading
# The company grew. We now have many customers and I can't handle the workload.
# We are now 5 developers doing exactly the same things.


def install_wordpress(customer):
    print("Start installation for", customer)
    sleep(3)
    print("All done for", customer)


def developer_day(customers):
    # But we now have to synchronize... a.k.a. bureaucracy
    lock = threading.Lock()

    def dev_day(id):
        print("Good morning from developer", id)
        # Yuck - I hate locks...
        lock.acquire()
        while customers:
            customer = customers.pop(0)
            lock.release()
            # My Python is less readable
            install_wordpress(customer)
            lock.acquire()
        lock.release()
        print("Bye from developer", id)
    # We go to work in the morning
    devs = [threading.Thread(target=dev_day, args=(i,)) for i in range(5)]
    [dev.start() for dev in devs]
    # we leave for the evening
    [dev.join() for dev in devs]


# We now get more done in the same time but our dev process got more complex.
# As we grew we spend more time managing queues than doing dev work. We even
# had occasional deadlocks when processes got extremely complex. The fact is
# that we are still mosly pressing buttons and waiting but now we also spend
# some time in meetings.
developer_day(["customer %d" % i for i in range(15)])
