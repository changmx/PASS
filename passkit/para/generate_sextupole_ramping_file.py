import numpy as np
import os
import csv


class SextRampingData:
    def __init__(self, startTurn, endTurn, k2):
        self.startTurn = startTurn
        self.endTurn = endTurn
        self.k2 = k2


def generate_sextupole_ramping_file_const(
    fileName="sextupole_ramping_file.csv", input_data=[]
):

    current_path, _ = os.path.split(__file__)
    parent_path = os.path.dirname(current_path)
    config_path = os.sep.join([parent_path, "para", fileName])
    print("The ramping file will be written to file: ", config_path)

    ###################################### Start generate ######################################

    output_data = []

    for data in input_data:
        for turn in range(data.startTurn, data.endTurn):
            output_data.append([turn, data.k2])

    with open(config_path, "w+", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(output_data)

    print("Success")


def generate_sextupole_ramping_file_linear(
    fileName="sextupole_ramping_file.csv",
    startTurn=0,
    endTurn=1,
    startValue=0,
    endValue=1,
    platformEnd=1,
):

    current_path, _ = os.path.split(__file__)
    parent_path = os.path.dirname(current_path)
    config_path = os.sep.join([parent_path, "para", fileName])
    print("The ramping file will be written to file: ", config_path)

    ###################################### Start generate ######################################

    output_data = []

    for turn in range(startTurn, endTurn):
        output_data.append(
            [turn, (endValue - startValue) / (endTurn - startTurn) * turn]
        )
    for turn in range(endTurn, platformEnd):
        output_data.append([turn, endValue])

    with open(config_path, "w+", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(output_data)

    print("Success")


if __name__ == "__main__":
    # generate_sextupole_ramping_file_const(
    #     "sf1_sta1_ramping.csv",
    #     [
    #         SextRampingData(1, 500, 0.5),
    #         SextRampingData(500, 1000, 1),
    #         SextRampingData(1000, 1500, 1.5),
    #         SextRampingData(1500, 2000, 2),
    #         SextRampingData(2000, 2500, 2.5),
    #         SextRampingData(2500, 3000, 3),
    #         SextRampingData(3000, 3500, 3.5),
    #         SextRampingData(3500, 4000, 4),
    #         SextRampingData(4000, 5000, 4.4),
    #     ],
    # )

    generate_sextupole_ramping_file_linear(
        "sf1_sta1_ramping.csv",
        startTurn=0,
        endTurn=4000,
        startValue=0,
        endValue=4.4,
        platformEnd=5000,
    )
