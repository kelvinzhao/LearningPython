# 尝试爬一下简书网站作者的收入
# url: https://www.jianshu.com/recommendations/users?page=5
# 通过查看页面，可分析到推荐作者列表中的作者主页链接可通过re获得；作者主页所需
# 信息可通过html树获得，因此使用requests-bs4-re技术框架
import requests
from bs4 import BeautifulSoup
import re
import openpyxl
import multiprocessing
# import time
import sys

# 增大recursion depth，否则会报错。
sys.setrecursionlimit(20000)
useragent = r"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
cookies = {}
# cookies已经存入文本，在此读取
# f = open(r'jianshucookies.txt', 'r')
with open('jianshucookies.txt', 'r') as f:
    for item in f.read().split(';'):
        key, value = item.strip().split('=', 1)
        cookies[key] = value


def getJSHTMLText(url):
    try:
        headers = {'user-agent': useragent}
        r = requests.get(url, timeout=30, headers=headers, cookies=cookies)
        r.raise_for_status()
        print(url + ' ' + str(r.status_code))
        r.encoding = r.apparent_encoding
        return r.text
    except Exception as e:
        print('error in getJSHTMLText', e)
        return ''


def getAuthorURLList(alt, html):
    # href="/users/080bb4eac1c9"
    baseurl = r'https://www.jianshu.com/u/'
    authorurls = re.findall(r'href\=\"\/users\/[0-9a-z]+\"', html)
    for item in authorurls:
        alt.append(baseurl + item.split(r'/')[2][:-1])


def getResultList(url):
    try:
        html = getJSHTMLText(url)
        soup = BeautifulSoup(html, 'html.parser')
        _name = soup.find('div', attrs={'class': 'title'})('a')[0].string
        _list = soup.find_all('div', attrs={'class': 'meta-block'})
        _infolist = []
        for item in _list:
            _infolist.append(item('p')[0].string)
        return {'name': _name, 'info': _infolist}
    except Exception as e:
        print('error in getResultList', e)
        return ''


def printResult(ilt):
    tplt = "{:16}\t{:5}\t{:5}\t{:5}\t{:5}\t{:5}\t{:5}"
    for info in ilt:
        if 'w' in info[6]:
            info[6] = str(eval(info[6][:-1]) * 10000)
        print(tplt.format(info[0], info[1], info[2],
                          info[3], info[4], info[5], info[6]))


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


if __name__ == '__main__':
    pageCount = 1
    authorURLList = []
    resultList = []
    # step1 循环两页推荐作者列表，获得作者主页链接列表
    for pageID in range(pageCount):
        url = 'https://www.jianshu.com/recommendations/users?page=' + \
                str(pageID + 1)
        html = getJSHTMLText(url)
        getAuthorURLList(authorURLList, html)

    # step2 循环列表，获得作者主页html text，爬取作者信息，包括关注、粉丝、文章、
    # 字数、收获喜欢、总资产等信息
    # 这里进程数为1，因为多了简书会返回429错误，“温和”的劝退
    pool = multiprocessing.Pool(1)
    resultList = pool.map(getResultList, authorURLList)

    # step3 输出结果列表
    resultValue = [["姓名", "关注", "粉丝", "文章", "字数", "喜欢", "资产"]]
    for one in resultList:
        resultValue += [[one['name']] + one['info']]

    print(resultValue)
    printResult(resultValue)
    writeExcel('jianshuauthors.xlsx', 'authors', resultValue)
