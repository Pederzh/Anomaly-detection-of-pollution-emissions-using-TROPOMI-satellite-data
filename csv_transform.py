location_names = ["Bering Strait", "Sabetta Port"]
location_name = location_names[1]
components = ["u", "v"] # x, y
component = components[0]

with open("Data/" + location_name + "/wind/wind-20190101_20211007-10m_" + component + "-500_600_700_800-OLD.csv", 'r') as f_in, open("data/" + location_name + "/wind/wind-20190101_20211007-10m_" + component + "-500_600_700_800.csv", 'w') as f_out:
    [f_out.write(','.join(line.split()) + '\n') for line in f_in]
    f_out.write(next(f_in))