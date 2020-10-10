# https://docs.scrapy.org/en/latest/intro/tutorial.html
import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
                'http://quotes.toscrape.com/page/1/',
                'http://quotes.toscrape.com/page/2/',
                ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

# 'start_requests' is default method that Scrapy generates 'scrapy.Request'
# objects from URLs, and 'start_urls' class attribute is the default URLs list.
# and 'parse' is default callback method.
# so 'start_request()' can be simplized as below.
#
#     name = "quotes"
#     start_urls = [
#                 'http://quotes.toscrape.com/page/1/',
#                 'http://quotes.toscrape.com/page/2/',
#                 ]

    def parse(self, response):
        # page = response.url.split("/")[-2]
        # filename = 'quote-%s.html' % page
        # with open(filename, 'wb') as f:
        #     f.write(response.body)
        # self.log('Saved file %s' % filename)
        for quote in response.css("div.quote"):
            yield {
                    'text': quote.css("span.text::text").get(),
                    'author': quote.css("small.author::text").get(),
                    'tags': quote.css("a.tag::text").getall(),
            }
        # add 'follow' to continue crawling next page
        # get url of next page
        # response.css("li.next a::attr(href)").get()
        # or
        # response.css("li.next a").attrib['href']
        # 'attrib' is a dict store all attribution in html tag
        next_page = response.css("li.next a").attrib['href']
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
            # A shortcut for creating Request objects by using `response.follow`
            # yield response.follow(next_page, callback=self.parse)
            # `response.follow` can deal with relative url, so no need urljoin
            # function
            # another shortcut by using `response.follow` is to pass <a> tag
            # directly in, and `response.follow` use href attribution automatically
            # such as:
            # for a in response.css("ul.pager a"):
            #    yield response.follow(a, callback=self.parse)
# check this out to explore how to use css combinators.
# https://www.w3.org/TR/2018/WD-selectors-4-20181121/#combinators
