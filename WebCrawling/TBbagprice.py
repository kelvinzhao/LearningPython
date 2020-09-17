# 淘宝比价平台爬虫
# step1 提交商品请求，循环获取页面
# step2 对于每个页面，提取商品名称和价格信息
# step3 将信息输出到屏幕上
# 通过查看淘宝页面网页代码，发现所需数据不是html代码，所以不使用bs4库，
# 而是采用requests-re技术路线
import requests
import re


def getTBHTMLText(url):
    f = open('./cookies.txt', 'r')
    cookies = {}
    for line in f.read().split(';'):
        name, value = line.strip().split('=', 1)
        cookies[name] = value
    f.close()
    try:
        r = requests.get(url, timeout=30, cookies=cookies)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except Exception:
        print("error in connecting")
        return ""


def parsePage(ilt, html):
    try:
        plt = re.findall(r'\"view_price\"\:\"[\d\.]*\"', html)
        tlt = re.findall(r'\"raw_title\"\:\".*?\"', html)
        for i in range(len(plt)):
            price = eval(plt[i].split(':')[1])
            title = eval(tlt[i].split(':')[1])
            ilt.append([price, title])
    except Exception:
        ilt = []


def printGoodsList(ilt):
    tplt = "{:4}\t{:5}\t{:16}"
    print(tplt.format("序号", "价格", "商品名称"))
    count = 0
    for g in ilt:
        count = count + 1
        print(tplt.format(count, g[0], g[1]))


def main():
    goods = '书包'
    depth = 2
    start_url = 'https://s.taobao.com/search?q=' + goods
    infoList = []
    for i in range(depth):
        try:
            url = start_url + '&s=' + str(i*44)
            html = getTBHTMLText(url)
            print(html)
            parsePage(infoList, html)
        except Exception:
            print("error occured")
            continue
    printGoodsList(infoList)


main()
