class Model:
    def __init__(self, index, name, accuracy, precision, recall, f1, model, y_pred):
        self.index = index
        self.name = name
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.model = model
        self.y_pred = y_pred

    # def __str__(self):
    #     return f"Model(index={self.index}, name='{self.name}', accuracy={self.accuracy}, precision={self.precision}, recall={self.recall}, f1={self.f1}, model={self.model}, y_pred={self.y_pred})"
