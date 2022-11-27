import pandas as pd
data={}
class Database:
    def create(self):  
        data = {'Name': ['Tom', 'Joseph', 'Krish', 'John'], 'Age': [20, 21, 19, 18]}  
        df = pd.DataFrame(data)
        print(df)  

    def read(self,file):
        print('read')
        excel = pd.read_excel(file)

    def update(self):
        data2 = {'Name': ['Tom', 'Joseph', 'Krish', 'John'], 'Age': [20, 21, 19, 18]}  
        data.update(data2)
        print(data2)

    def delete(self):
        data.clear()
        print(data)
        