import csv


class ExemplarProvider:
    def __init__(self, filename):
        self.file = open(filename)
        self.reader = csv.reader(self.file)
        self.data_list = list(self.reader)
        self.N = len(self.data_list)
        self.file.seek(0)

    def read_next_line(self):
        print(next(self.reader))

    def get_file_size(self):
        return self.N