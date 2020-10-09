import scrapy


class AuthorspiderSpider(scrapy.Spider):
    name = 'AuthorSpider'
    allowed_domains = ['jianshu.com']
    # cookies = {}
    # with open('jianshucookies.txt', 'r') as f:
    #     for item in f.read().split(';'):
    #         key, value = item.strip().split('=', 1)
    #         cookies[key] = value

    # headers = {}
    # headers['user-agent'] = r"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6)"

    def start_requests(self):
        pageCount = 10
        for pageID in range(pageCount):
            url = 'https://www.jianshu.com/recommendations/users?page=' + \
                    str(pageID + 1)
            yield scrapy.Request(url=url)

    def parse(self, response):
        # parse response to get author list
        yield from response.follow_all(css='div.wrap > a', callback=self.parse_author)

    def parse_author(self, response):
        authorinfo = response.css("div.info * p::text").getall()
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
