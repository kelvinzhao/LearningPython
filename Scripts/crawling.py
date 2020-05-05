import urllib.request
if __name__ == "__main__":
    httpshandler = urllib.request.HTTPSHandler()
    opener = urllib.request.build_opener(httpshandler)
    urllib.request.install_opener(opener)
    response = urllib.request.urlopen('https://kelvinzhao.me')
    html = response.read()
    html = html.decode('utf-8')
    print(html)
