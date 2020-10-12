import scrapy


class HotSpider(scrapy.Spider):
    name = 'hot'
    allowed_domains = ['s.weibo.com']
    start_urls = ['https://s.weibo.com/top/summary?cate=realtimehot']

    def parse(self, response):
        for hid, hotword, redu in zip(
                range(0, 51),
                response.css("tbody * a::text").getall(),
                response.css("tbody * span::text").getall(),
                ):
            yield {
                    'hid': hid,
                    'hotword': hotword,
                    'redu': redu,
                    }
