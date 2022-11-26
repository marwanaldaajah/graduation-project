import requests
import csv

url="https://en.wikipedia.org/api/rest_v1/page/title/Earth"


 headers ={
    "accept":"application",
    "content_type":"application/json"
 }
response = requests.request("GET",url,headers=headers,data={})
myjson = response.json()
ourdata=[]


for x in myjson["data"]:
    listing =[x["name"],x["subject"]]
    ourdata.append(listing)
    
print (myjson)    