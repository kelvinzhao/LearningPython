# import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class EasySpider(CrawlSpider):
    name = 'easy'
    allowed_domains = ['jianshu.com']
    start_urls = ['https://www.jianshu.com/recommendations/users?page=1',
                  'https://www.jianshu.com/recommendations/users?page=2']

    rules = (
        # 以下两种写法均可
        # Rule(LinkExtractor(restrict_xpaths='//div[has-class("wrap")]/a[1]'), callback='parse_item'),
        Rule(LinkExtractor(allow=r'/users/'), callback='parse_item'),
    )

    def parse_item(self, response):
        authorinfo = response.xpath('//div[has-class("info")]//p/text()').getall()
        if 'w' in authorinfo[5]:
            authorinfo[5] = str(int(eval(authorinfo[5][:-1])) * 10000)
        yield {
                'name': response.css("a.name::text").get(),
                'follow': authorinfo[0],
                'follower': authorinfo[1],
                'articles': authorinfo[2],
                'words': authorinfo[3],
                'likes': authorinfo[4],
                'rewards': authorinfo[5],
                }
