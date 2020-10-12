# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
# from itemadapter import ItemAdapter
import openpyxl
from datetime import datetime


class WeibohotPipeline:
    def open_spider(self, spider):
        self.wb = openpyxl.Workbook()
        self.sheet = self.wb.create_sheet(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        self.sheet.append(['排名', '热搜', '热度'])

    def close_spider(self, spider):
        pass

    def process_item(self, item, spider):
        self.sheet.append([item['hid'], item['hotword'], item['redu']])
        self.wb.save("weibohot.xlsx")
        return item
