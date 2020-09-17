# 中国大学排名定向爬虫实例
# 输入，排名URL
# 输出，排名、大学名称、总分在屏幕输出
# step1 : getHTMLText()
# step2 : fillUnivList()
# step3 : printUnivList()
import bs4
import requests
from bs4 import BeautifulSoup


def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except Exception:
        return ""


def fillNuvList(ulist, html):
    soup = BeautifulSoup(html, 'html.parser')
    for tr in soup.find('tbody', 'hidden_zhpm').children:
        if isinstance(tr, bs4.element.Tag):
            tds = tr('td')
            ulist.append([tds[0].string, tds[1].string, tds[4].string])


def printUnivList(ulist, num):
    tplt = "{0:^5}\t{1:{3}^10}\t{2:^5}"
    print(tplt.format("Range", "University", "Score", chr(12288)))
    for i in range(num):
        u = ulist[i]
        print(tplt.format(u[0], u[1], u[2], chr(12288)))


def main():
    uinfo = []
    url = 'http://www.zuihaodaxue.com/zuihaodaxuepaiming-zongbang-2020.html'
    html = getHTMLText(url)
    fillNuvList(uinfo, html)
    printUnivList(uinfo, 20)


main()
