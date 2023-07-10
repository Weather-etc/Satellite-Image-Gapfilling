"""
This is the spider to get tiff image automatically from the website:
https://products.coastalscience.noaa.gov/habs_explorer/
index.php?path=RUIvWnB3dWJmS3RvNXlWcjF4a1hLM1B0eERkak1wT2hueTFPRjFMSzVyQmJYMjVpQ2NmUz
k5eVllQlVQd1ZiTw==&uri=VWtuM1UzbVNVN0RsZzJMeTJvNlNpM29OalF0WTFQQjVZVnpuS3o5bnh1Ym0vYWhtWEh4ck1hREVUamE4SDZ0M2tsd1M
"""
import sys
import re
import requests
from lxml import etree

URLs = "https://www.baidu.com"
URL = "https://products.coastalscience.noaa.gov/habs_explorer/index.php?path=RUIvWnB3dWJmS3RvNXlWcjF4a1hLM1B0eERkak1wT2hueTFPRjFMSzVyQmJYMjVpQ2NmUzk5eVllQlVQd1ZiTw==&uri=VWtuM1UzbVNVN0RsZzJMeTJvNlNpM29OalF0WTFQQjVZVnpuS3o5bnh1Ym0vYWhtWEh4ck1hREVUamE4SDZ0M2tsd1M"
header = {
    'user-agent':
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.58",
    'Connection': "keep-alive"
}
# My proxy server only support http protocol
proxies = {'http':
               "http://127.0.0.1:7890",
           'https':
               "http://127.0.0.1:7890"
           }

savePath = '../data/rawTif/'

response = requests.get(url=URL, headers=header, proxies=proxies)
html = etree.HTML(response.text)
divs = html.xpath('//div[@class="container"]/div[@class="row"]/section[contains(text(), "CIcyano")]')
for div in divs:
    imgName = div.xpath('./text()')[0][20:24]
    imgUrl = div.xpath('./a/@href')[0]

    try:
        imgResponse = requests.get(url=imgUrl, headers=header, proxies=proxies)
        imgBi = imgResponse.content
        imgPath = savePath + imgName + '.tif'
        with open(imgPath, 'wb') as f:
            f.write(imgBi)
    except Exception as ex:
        print("ERROR:")
        print(ex.with_traceback(sys.exc_info()[2]))
