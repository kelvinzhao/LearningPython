import scrapy


class ZhongkaoSpider(scrapy.Spider):
    name = 'zhongkao'
    allowed_domains = ['bjeea.cn']
    start_urls = ['https://www.bjeea.cn/html/zkzz/tzgg/2020/0727/76251.html']

    def parse(self, response):
        yield from response.follow_all(css='table * a', callback=self.distrct_parse)

    def distrct_parse(self, response):
        trlist = response.css("table tbody tr")
        title = trlist[0].css('td::text').get()
        _tmplist = []
        for tr in trlist[1:]:
            _tmplist.append([
                tr.css('td::text')[0].get(),
                tr.css('td::text')[1].get(),
                tr.css('td::text')[2].get(),
                    ])
        yield {
                'name': title,
                'scorelist': _tmplist,
                }
