class Model:
    def __init__(self,index, name, accuracy, precision, recall, f1,model):
        self.index = index
        self.name = name
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.model=model
