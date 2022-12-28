import csv

def print_sol_summary_csv(kernel_id_characteristics):
    #print header
    header = "kernel_name, block_size, grid_size, "
    metric_header = ""
    rows = []
    for kernel_id, kernel_characteristics in kernel_id_characteristics.items():
        #print("kernel id: ", kernel_id)
        #print("kernel name: ", kernel_characteristics["kernel_name"])
        #print("block size: ", kernel_characteristics["block_size"])
        #print("grid size: ", kernel_characteristics["grid_size"])
        metrics_string = ""
        metric_name_string = ""
        for metric_name, metric_value_unit in kernel_characteristics.items():
            if metric_name in ["kernel_name", "block_size", "grid_size"]:
                continue
            metric_name_string += metric_name + ","
            #print(metric_name, ": ", metric_value_unit[0], metric_value_unit[1])
            metrics_string += (metric_value_unit[0] + " " + metric_value_unit[1]) + ", "
        pass
        if metric_header == "":
            metric_header = metric_name_string
        else:
            if metric_header != metric_name_string:
                print("metric header mismatch"+kernel_id +"("+ metric_header + " vs " + metric_name_string)
        rows.append(kernel_id + ", " + kernel_characteristics["kernel_name"] + ", " + kernel_characteristics["block_size"] + ", "+ kernel_characteristics["grid_size"]+", "+metrics_string)

    print(header + metric_header)
    for row in rows:
        print(row)
    pass

def extract_sol_from_csv(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        kernel_id_characteristics = dict()
        for idx_row, row in enumerate(reader):
            if idx_row == 0: # extract header
                header_to_col_idx = {header: idx_col for idx_col, header in enumerate(row)}
                continue
            # extract ID, Kernel Name, Block Size, Grid Size, Section Name == "GPU Speed Of Light Throughput", Metric Name, metric unit, matric value
            # construct a dictionary for each unique id
            if row[header_to_col_idx["Section Name"]] == "GPU Speed Of Light Throughput":
                kernel_id = row[header_to_col_idx["ID"]]
                kernel_name = row[header_to_col_idx["Kernel Name"]]
                block_size = row[header_to_col_idx["Block Size"]]
                grid_size = row[header_to_col_idx["Grid Size"]]
                metric_name = row[header_to_col_idx["Metric Name"]]
                metric_unit = row[header_to_col_idx["Metric Unit"]]
                metric_value = row[header_to_col_idx["Metric Value"]]
                if kernel_id not in kernel_id_characteristics:
                    kernel_id_characteristics[kernel_id] = dict()
                    kernel_id_characteristics[kernel_id]["kernel_name"] = kernel_name
                    kernel_id_characteristics[kernel_id]["block_size"] = block_size
                    kernel_id_characteristics[kernel_id]["grid_size"] = grid_size
                kernel_id_characteristics[kernel_id][metric_name] = (metric_value, metric_unit)
            pass
        pass
    pass
    return kernel_id_characteristics

if __name__ == "__main__":
    kernel_id_characteristics = extract_sol_from_csv("myhgt_sol.csv")
    print_sol_summary_csv(kernel_id_characteristics)
    pass