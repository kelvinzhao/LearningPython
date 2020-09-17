# %%
import requests
from bs4 import BeautifulSoup

r = requests.get('http://python123.io/ws/demo.html')
print(r.status_code)
demo = r.text
# %%
soup = BeautifulSoup(demo, "html.parser")
print(soup.prettify())
# %%
soup.a.name
soup.a.parent.name
soup.a.parent.parent.name
# %%
tag = soup.a
tag.attrs
tag.attrs['href']
type(tag)
# %%
newsoup = BeautifulSoup('<b><!-- This is a comment --></b><p>This is not a comment</p>', 'html.parser')
# %%
soup.find_all(id='link1')
# %%
import re
string = 'BIT100081 TSU100084'
m = re.search(r'[1-9]\d{5}', string)
if m:
    print(m.group(0))

m.string
m.re
m.pos
m.endpos
m.group(0)
m.start()
m.end()
m.span()
re.sub(r'[1-9]\d{5}', '-', string)
