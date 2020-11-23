# ~*~ Twisted - A Python tale ~*~
# Part 1
from time import sleep
# Hello, I'm a developer and I mainly setup Wordpress.


def install_wordpress(customer):
    # Our hosting company Threads Ltd. is sucks. I start installation and ...
    print("Start installation for", customer)
    # ...then wait till the installation finishes successfully. It is boring
    # and I'm spending most of my time waiting while consuming resources
    # (memory and some CPU cycles). It's because the process is *BLOCKING*.
    sleep(3)
    print("All done for", customer)


# I do this all day long for our customers
def developer_day(customers):
    for customer in customers:
        install_wordpress(customer)


developer_day(['Bill', 'Elon', 'Steve', 'Mark'])
