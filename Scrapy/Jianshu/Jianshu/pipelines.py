# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
# from itemadapter import ItemAdapter
# import json
import openpyxl


class JianshuAuthorPipeline:

    def open_spider(self, spider):
        # self.file = open("jianshuauthor.jl", "w")
        self.workbook = openpyxl.Workbook()
        self.sheet = self.workbook.active
        self.sheet.append(["姓名", "关注", "粉丝", "文章", "字数", "喜欢", "资产"])

    def close_spider(self, spider):
        # self.file.close()
        # self.workbook.save("jianshuAuthor.xlsx")
        pass

    def process_item(self, item, spider):
        line = [
                item['name'],
                item['follow'],
                item['follower'],
                item['articles'],
                item['words'],
                item['likes'],
                item['rewards']
                ]
        self.sheet.append(line)
        self.workbook.save("jianshuAuthor.xlsx")
        return item