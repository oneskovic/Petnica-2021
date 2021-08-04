from copy import deepcopy
class Logger:
    def __init__(self):
        self.logged_values = dict()
        self.logged_msgs = dict()

    def log_value(self, name, generation_number, value):
        if name not in self.logged_values:
            self.logged_values[name] = dict()
        self.logged_values[name][generation_number] = deepcopy(value)

    def log_msg(self, msg, generation_number):
        self.logged_msgs[generation_number] = msg

    def get_values(self):
        return self.logged_values
