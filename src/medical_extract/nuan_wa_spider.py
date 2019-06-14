# coding=utf-8

import json
import urllib.request

url_disease_detail = 'https://med-askbob.pingan.com/pedia/disease/detail?key={}'
headers = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 12_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/7.0.4(0x17000428) NetType/WIFI Language/zh_CN",
    "authentication":"eyJhbGciOiJIUzUxMiJ9.eyJhcHBsaWNhdGlvbkFjY291bnRJbmZvIjp7ImlkIjoxMDM0OSwiY2hhbm5lbElkIjoiMTEwMDQ5MDAwMCIsImluc3RpdHV0aW9uSWQiOiIxMjQ0NDQwMzAwMDAzMzEwMDAwMDAwIiwicm9sZSI6MSwic291cmNlIjoxLCJzZXNzaW9uVHlwZSI6IndlY2hhdCIsImlzQXV0b0xvZ2luIjpmYWxzZSwiY29tbW9uVXNlcklkIjoxMTMxLCJwYXltZW50TGV2ZWwiOm51bGx9LCJleHAiOjE1NzU5NTY0MTV9.vOQ7vA_u3noTj7X2rEGTrC9Dre_bG-AF2dOroxdDkR26PR_ErnozaL0e_2vpuTEDhLNIfRUn9ya5CDvBlDzQxQ"
           }

req = urllib.request.Request(url_disease_detail.format('a44a0b299afcf3e0bca6651e84fd0e57'), headers=headers)
response = urllib.request.urlopen(req)
the_page = response.read()
result = str(the_page, encoding="utf-8")
cl = json.loads(result)
print(cl)
