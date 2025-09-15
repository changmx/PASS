import pandas as pd
import os


def merge_txt_to_csv(file_a, file_b, file_c, file_d, output_csv):

    try:
        # 读取四个TXT文件
        data_a = pd.read_csv(file_a, header=None, names=["harmonic"])
        data_b = pd.read_csv(file_b, header=None, names=["voltage"])
        data_c = pd.read_csv(file_c, header=None, names=["phis"])
        data_d = pd.read_csv(file_d, header=None, names=["phi_offset"])

        # 合并数据
        merged_data = pd.concat([data_a, data_b * 1e6, data_c, data_d], axis=1)

        # 保存为CSV文件
        merged_data.to_csv(output_csv, index=False)

        print(f"成功合并文件到 {output_csv}")
        print(f"合并后的数据形状: {merged_data.shape}")

        return True

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        return False


if __name__ == "__main__":
    merge_txt_to_csv(
        r"D:\PASS\para\rf\h_interpolation.txt",
        r"D:\PASS\para\rf\v_interpolation.txt",
        r"D:\PASS\para\rf\p_interpolation.txt",
        r"D:\PASS\para\rf\po_interpolation.txt",
        r"D:\PASS\para\rf_data.csv",
    )
