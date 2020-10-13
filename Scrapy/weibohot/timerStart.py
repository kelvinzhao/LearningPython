from scrapy import cmdline
import datetime
import time
import os

checkFile = r"isRunning.txt"
i = 0

while True:
    isRunning = os.path.isfile(checkFile)
    if not isRunning:
        cmdline.execute('scrapy crawl hot'.split())
    else:
        print(f"At time: {datetime.datetime.now()}, hot spider is running now, sleep to wait.")
    i += 1
    time.sleep(120)
    if i >= 10:
        break
