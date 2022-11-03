# from urllib.request import urlopen
#
# url = "http://www.baidu.com/"
# resp = urlopen(url)
#
# with open("test.html", mode="w",encoding='utf-8') as f:
#     f.write(resp.read().decode("utf-8"))
# print("over!")


###### 2 request
import requests

# url = "https://sogou.com/web?query=jack"
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
# }
# resp = requests.get(url, headers=headers)
# print(resp)
# print(resp.text)

# url = "https://fanyi.baidu.com/sug"
# data = {
#     "kw": "dog"
# }
# resp = requests.post(url, data=data)
# print(resp.json())

url = "https://movie.douban.com/j/chart/top_list"
params = {
    "type": "24",
    "interval_id": "100:90",
    "action": "",
    "start": "0",
    "limit": "20",
}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
}
resp = requests.get(url=url, params=params, headers=headers)

print(resp.json())
resp.close()
