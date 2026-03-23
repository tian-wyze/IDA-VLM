import pandas as pd
import sys

if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        print("Usage: python cal_acc.py result.csv")
        sys.exit(1)

    file = sys.argv[1]
    result = pd.read_csv(file)

    # calculate the acc based on the label and prediction columns
    acc = (result['label'] == result['prediction']).mean() * 100
    print(f'Acc: {round(acc, 1)}')


