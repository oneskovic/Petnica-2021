from copy import deepcopy
class Logger:
    def __init__(self, detalied=False):
        self.logged_values = dict()
        self.logged_msgs = dict()
        self.detailed = detalied

    def log_value(self, name, generation_number, value):
        if name not in self.logged_values:
            self.logged_values[name] = dict()
        self.logged_values[name][generation_number] = deepcopy(value)

    def log_msg(self, msg, generation_number):
        if generation_number not in self.logged_msgs:
            self.logged_msgs[generation_number] = list()
        self.logged_msgs[generation_number].append(msg)


    def get_values(self):
        return self.logged_values
