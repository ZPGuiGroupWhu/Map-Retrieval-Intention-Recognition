import json


def save_as_json(data, path):
    data_json = json.dumps(data)
    data_file = open(path, "w+")
    data_file.write(data_json)
    data_file.close()


def load_json(path):
    data_file = open(path)
    data = json.load(data_file)
    data_file.close()
    return data


def save_as_csv(data, path):
    data_file = open(path, "w+")
    for tmp_line_data in data:
        tmp_line_str = ",".join([str(x) for x in tmp_line_data]) + "\n"
        data_file.write(tmp_line_str)
    data_file.close()


def load_csv(path):
    data_file = open(path,encoding='utf-8')
    data = []
    for tmp_line_str in data_file.readlines():
        tmp_line_data = tmp_line_str.strip().split(",")
        data.append(tmp_line_data)
    data_file.close()
    return data
