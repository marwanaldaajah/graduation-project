import requests 
subject = {"subjectName":"subject"}
r.requests.post("https://httpbin.org/post",data=subject)
print(r.text)