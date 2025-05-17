import json

class Timetable:
    def __init__(self):
        self.data = {}

    def load_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def save_to_json(self, json_path):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4) 