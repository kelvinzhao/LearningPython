# ~*~ Twisted - A python tale ~*~
# Part 4 (Better version for part 3)
# For years we thought this was all there was ... we kept hiring more developers,
# more managers and buying servers. We were trying harder optimising processes
# and fire-fighting while getting mediocre performance in return.
# Till luckily one day our hosting company decided to increase their fees and
# we decide to switch to Twisted Ltd.!
from twisted.internet import reactor
from twisted.internet import defer
from twisted.internet import task


# 关于@defer.inlineCallbacks装饰符
# 参看: http://timd.cn/python/scrapy/inlinecallbacks/
@defer.inlineCallbacks
def inline_install(customer):
    print("Scheduling: Installation for", customer)
    yield task.deferLater(reactor, 3, lambda: None)
    print("Callback: Finished installation for", customer)
    print("All done for", customer)

# Yes, we don't need many developers anymore or any synchronization.
# ~~ Super-powered Twisted developer ~~


def twisted_developer_day(customers):
    print("Good morning from Twisted developer")
    # Here's what has to be done today
    work = [inline_install(customer) for customer in customers]
    # Turn off the lights when done
    join = defer.DeferredList(work)
    join.addCallback(lambda _: reactor.stop())
    #
    print("Bye from Twisted developer!")


# Even his day is particularly short!
twisted_developer_day(["Customer %d" % i for i in range(15)])

# Reactor, our secretary uses the CRM and follows-up on events!
reactor.run()
