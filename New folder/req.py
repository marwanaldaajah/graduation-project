import requests
subject = {"subjectName":"subject"}
r.requests.get("https://httpbin.org/get",params=subject)
print(r.text)