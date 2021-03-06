import scrapy


class HotSpider(scrapy.Spider):
    name = 'hot'
    allowed_domains = ['s.weibo.com']
    start_urls = ['https://s.weibo.com/top/summary?cate=realtimehot']

    def parse(self, response):
        for hid, hotword, redu in zip(
                range(1, 51),
                response.css("td.td-01.ranktop + td.td-02 a:not([href_to])::text").getall(),
                response.css("td.td-01.ranktop + td.td-02 a:not([href_to]) + span::text").getall(),
                ):
            yield {
                    'hid': hid,
                    'hotword': hotword,
                    'redu': redu,
                    }
