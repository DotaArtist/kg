# coding=utf-8

import json
import urllib.request

url_disease_detail = 'https://med-askbob.pingan.com/pedia/disease/detail?key={}'
headers = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 12_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/7.0.4(0x17000428) NetType/WIFI Language/zh_CN",
    "authentication":"eyJhbGciOiJIUzUxMiJ9.eyJhcHBsaWNhdGlvbkFjY291bnRJbmZvIjp7ImlkIjoxMDM0OSwiY2hhbm5lbElkIjoiMTEwMDQ5MDAwMCIsImluc3RpdHV0aW9uSWQiOiIxMjQ0NDQwMzAwMDAzMzEwMDAwMDAwIiwicm9sZSI6MSwic291cmNlIjoxLCJzZXNzaW9uVHlwZSI6IndlY2hhdCIsImlzQXV0b0xvZ2luIjpmYWxzZSwiY29tbW9uVXNlcklkIjoxMTMxLCJwYXltZW50TGV2ZWwiOm51bGx9LCJleHAiOjE1NzU2OTc3MDZ9.JJet9uyUz4gegW_sVjNWDekHg6zhARlBAve6YLUaHevXHgFm1eMgN76hX7o49mxtbO9a2RvjO1j-tVdBqdSi1A"
           }

req = urllib.request.Request(url_disease_detail.format('a44a0b299afcf3e0bca6651e84fd0e57'), headers=headers)
response = urllib.request.urlopen(req)
the_page = response.read()
result = str(the_page, encoding="utf-8")
cl = json.loads(result)
print(cl)
