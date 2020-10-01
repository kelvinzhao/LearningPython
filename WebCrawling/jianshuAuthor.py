# 尝试爬一下简书网站作者的收入
# url: https://www.jianshu.com/recommendations/users?page=5
# 通过查看页面，可分析到推荐作者列表中的作者主页链接可通过re获得；作者主页所需
# 信息可通过html树获得，因此使用requests-bs4-re技术框架
import requests
from bs4 import BeautifulSoup
import re
import openpyxl


def getJSHTMLText(url, cookies, useragent):
    headers = {'user-agent': useragent}
    try:
        r = requests.get(url, timeout=30, headers=headers, cookies=cookies)
        r.raise_for_status()
        # print(url + ' ' + str(r.status_code))
        r.encoding = r.apparent_encoding
        return r.text
    except Exception:
        print('error in getJSHTMLText')
        return ''


def getAuthorURLList(alt, html):
    # href="/users/080bb4eac1c9"
    baseurl = r'https://www.jianshu.com/u/'
    authorurls = re.findall(r'href\=\"\/users\/[0-9a-z]+\"', html)
    for item in authorurls:
        alt.append(baseurl + item.split(r'/')[2][:-1])


def getResultList(ilt, html):
    soup = BeautifulSoup(html, 'html.parser')
    _name = soup.find('div', attrs={'class': 'title'})('a')[0].string
    _list = soup.find_all('div', attrs={'class': 'meta-block'})
    _infolist = []
    """
    <div class="meta-block">
        <a href="/users/51b4ef597b53/following">
            <p>11</p>
            关注 <i class="iconfont ic-arrow"></i>
        </a>
    </div>
    """
    for item in _list:
        _infolist.append(item('p')[0].string)
    ilt[_name] = _infolist


def printResult(ilt):
    tplt = "{:16}\t{:5}\t{:5}\t{:5}\t{:5}\t{:5}\t{:5}"
    print(tplt.format('姓名', '关注', '粉丝', '文章', '字数', '喜欢', '资产'))
    for name, info in ilt.items():
        if 'w' in info[5]:
            info[5] = str(eval(info[5][:-1]) * 10000)
        print(tplt.format(name, info[0], info[1], info[2],
                          info[3], info[4], info[5]))


def writeExcel(path, sheet_name, value):
    index = len(value)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    for i in range(index):
        for j in range(len(value[i])):
            sheet.cell(row=i+1, column=j+1, value=str(value[i][j]))
    workbook.save(path)
    print("write in excel files, done!")


def main():
    useragent = r"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6)"
    cookies = {}
    # cookies已经存入文本，在此读取
    # f = open(r'jianshucookies.txt', 'r')
    with open('jianshucookies.txt', 'r') as f:
        for item in f.read().split(';'):
            key, value = item.strip().split('=', 1)
            cookies[key] = value
    pageCount = 1
    authorURLList = []
    resultDict = {}
    # step1 循环两页推荐作者列表，获得作者主页链接列表
    for pageID in range(pageCount):
        url = 'https://www.jianshu.com/recommendations/users?page=' + \
                str(pageID + 1)
        try:
            html = getJSHTMLText(url, cookies, useragent)
            getAuthorURLList(authorURLList, html)
        except Exception:
            print('error in getAuthorURLList')
    # step2 循环列表，获得作者主页html text，爬取作者信息，包括关注、粉丝、文章、
    # 字数、收获喜欢、总资产等信息
    count = 0
    for url in authorURLList:
        try:
            html = getJSHTMLText(url, cookies, useragent)
            getResultList(resultDict, html)
            count += 1
            print('\r当前进度：{:.2f}%'.format(count*100/len(authorURLList)), end='')
        except Exception:
            print('error in getResultList')
    print('')
    # step3 输出结果列表
    printResult(resultDict)
    resultValue = [["姓名", "关注", "粉丝", "文章", "字数", "喜欢", "资产"]]
    for key, value in resultDict.items():
        _list = [key] + value
        resultValue.append(_list)
    writeExcel('jianshuauthors.xlsx', 'authors', resultValue)


main()
