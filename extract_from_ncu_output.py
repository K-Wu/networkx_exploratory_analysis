import csv
import os
import xlsxwriter


def extract_from_all_sol_files(folder_path):
    models = ["HGT.train", "RGAT.train"]
    compact_flag = ["--compact_as_of_node_flag", ""]
    mul_flag = ["--multiply_among_weights_first_flag", ""]
    datasets = ["aifb", "mutag", "bgs", "am", "mag", "wikikg2", "fb15k", "biokg"]
    # HGT C0 M0 starts from

    # deal with rgat and hgt
    kernel_id_characteristics_all = dict()
    for model in models:
        kernel_id_characteristics_all[model] = dict()
        for cf in compact_flag:
            kernel_id_characteristics_all[model][cf] = dict()
            for mf in mul_flag:
                kernel_id_characteristics_all[model][cf][mf] = dict()
    kernel_id_characteristics_all["RGCNSingleLayer"] = dict()

    for model in models:
        for cf in compact_flag:
            for mf in mul_flag:
                for dataset in datasets:
                    csv_file = os.path.join(
                        folder_path, "{}.{}.{}.{}.log".format(model, dataset, mf, cf)
                    )
                    kernel_id_characteristics = extract_sol_from_csv(csv_file)
                    kernel_id_characteristics_all[model][cf][mf][
                        dataset
                    ] = kernel_id_characteristics
                    # for kernel_id, characteristics in kernel_id_characteristics.items():
                    #     print("kernel_id: {}, characteristics: {}".format(kernel_id, characteristics))
                    #     pass
                    # pass

    # deal with rgcnsinglelayer
    for dataset in datasets:
        csv_file = os.path.join(folder_path, "RGCNSingleLayer.{}.log".format(dataset))
        kernel_id_characteristics = extract_sol_from_csv(csv_file)
        kernel_id_characteristics_all["RGCNSingleLayer"][
            dataset
        ] = kernel_id_characteristics
        # for kernel_id, characteristics in kernel_id_characteristics.items():
        #     print("kernel_id: {}, characteristics: {}".format(kernel_id, characteristics))
        #     pass
        # pass
    return kernel_id_characteristics_all


def print_all_sol_files_to_xlsx(kernel_id_characteristics_all):
    models = ["HGT.train", "RGAT.train"]
    compact_flag = ["--compact_as_of_node_flag", ""]
    mul_flag = ["--multiply_among_weights_first_flag", ""]
    datasets = ["aifb", "mutag", "bgs", "am", "mag", "wikikg2", "fb15k", "biokg"]
    # print each model to a xlsx file in which each dataset is a sheet
    for model in models:
        workbook = xlsxwriter.Workbook("{}.xlsx".format(model))
        for dataset in datasets:
            worksheet = workbook.add_worksheet(dataset)
            curr_row_idx = 0
            for cf in compact_flag:
                for mf in mul_flag:
                    # write flags
                    worksheet.write(curr_row_idx + 0, 0, "compact flag: {}".format(cf))
                    worksheet.write(curr_row_idx + 1, 0, "multiply flag: {}".format(mf))
                    curr_row_idx += 2
                    rows = get_sol_summary_sheet_rows(
                        kernel_id_characteristics_all[model][cf][mf][dataset]
                    )
                    for row in rows:
                        for col_idx, cell in enumerate(row):
                            worksheet.write(curr_row_idx, col_idx, cell)
                        curr_row_idx += 1
        workbook.close()

    # print rgcnsinglelayer to a xlsx file in which each dataset is a sheet
    workbook = xlsxwriter.Workbook("RGCNSingleLayer.xlsx")
    for dataset in datasets:
        worksheet = workbook.add_worksheet(dataset)
        curr_row_idx = 0
        rows = get_sol_summary_sheet_rows(
            kernel_id_characteristics_all["RGCNSingleLayer"][dataset]
        )
        for row in rows:
            for col_idx, cell in enumerate(row):
                worksheet.write(curr_row_idx, col_idx, cell)
            curr_row_idx += 1
    workbook.close()


def get_sol_summary_sheet_rows(kernel_id_characteristics):
    # print header
    header = ["kernel_id", "kernel_name"]
    metric_header = []
    rows = []
    for kernel_id, kernel_characteristics in kernel_id_characteristics.items():
        # print("kernel id: ", kernel_id)
        # print("kernel name: ", kernel_characteristics["kernel_name"])
        # print("block size: ", kernel_characteristics["block_size"])
        # print("grid size: ", kernel_characteristics["grid_size"])
        metrics_strings = []
        metric_name_strings = []
        for metric_name, metric_value_unit in kernel_characteristics.items():
            if metric_name in ["kernel_name", "block_size", "grid_size"]:
                continue
            metric_name_strings.append(metric_name)
            # print(metric_name, ": ", metric_value_unit[0], metric_value_unit[1])
            metrics_strings.append((metric_value_unit[0] + " " + metric_value_unit[1]))
        pass
        if len(metric_header) == 0:
            metric_header = metric_name_strings
        else:
            if metric_header != metric_name_strings:
                print(
                    "metric header mismatch"
                    + kernel_id
                    + "("
                    + metric_header
                    + " vs "
                    + metric_name_strings
                )
        rows.append(
            [kernel_id, kernel_characteristics["kernel_name"]] + metrics_strings
        )
    rows = [header + metric_header] + rows
    return rows


def print_sol_summary_csv(kernel_id_characteristics):
    rows = get_sol_summary_sheet_rows(kernel_id_characteristics)
    for row in rows:
        print(row.join(","))
    pass


def extract_sol_from_csv(csv_file):
    with open(csv_file, "r") as f:
        # skip if first line starts with =Error=
        first_line = f.readline()
        if first_line.strip().startswith("==ERROR=="):
            return {}
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        kernel_id_characteristics = dict()
        for idx_row, row in enumerate(reader):
            if idx_row == 0:  # extract header
                header_to_col_idx = {
                    header: idx_col for idx_col, header in enumerate(row)
                }
                continue
            # extract ID, Kernel Name, Block Size, Grid Size, Section Name == "GPU Speed Of Light Throughput", Metric Name, metric unit, matric value
            # construct a dictionary for each unique id
            if (
                row[header_to_col_idx["Section Name"]]
                == "GPU Speed Of Light Throughput"
            ):
                kernel_id = row[header_to_col_idx["ID"]]
                kernel_name = row[header_to_col_idx["Kernel Name"]]
                # block_size = row[header_to_col_idx["Block Size"]]
                # grid_size = row[header_to_col_idx["Grid Size"]]
                metric_name = row[header_to_col_idx["Metric Name"]]
                metric_unit = row[header_to_col_idx["Metric Unit"]]
                metric_value = row[header_to_col_idx["Metric Value"]]
                if kernel_id not in kernel_id_characteristics:
                    kernel_id_characteristics[kernel_id] = dict()
                    kernel_id_characteristics[kernel_id]["kernel_name"] = kernel_name
                    # kernel_id_characteristics[kernel_id]["block_size"] = block_size
                    # kernel_id_characteristics[kernel_id]["grid_size"] = grid_size
                kernel_id_characteristics[kernel_id][metric_name] = (
                    metric_value,
                    metric_unit,
                )
            pass
        pass
    pass
    return kernel_id_characteristics


def test_load():
    kernel_id_characteristics = extract_sol_from_csv("myhgt_sol.csv")
    print_sol_summary_csv(kernel_id_characteristics)


def test_differential():
    kernel_id_characteristics_all = extract_from_all_sol_files(
        os.path.join(".", "ncu_breakdown_with_warm_up_5")
    )
    kernel_id_characteristics_all_no_warm_up = extract_from_all_sol_files(
        os.path.join(".", "ncu_breakdown_without_warm_up_or_print")
    )
    # we can use the kernel id counts in kernel_id_characteristics_all_no_warm_up to figure out how many kernels to be obtained from kernel_id_characteristics_all
    # the kernel should be those in the end in each of the profile runs


if __name__ == "__main__":
    kernel_id_characteristics_all_no_warm_up = extract_from_all_sol_files(
        os.path.join(".", "ncu_breakdown_without_warm_up_or_print")
    )
    print_all_sol_files_to_xlsx(kernel_id_characteristics_all_no_warm_up)
    pass
