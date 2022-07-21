import csv
import ml


def run(csv_file):
    with open(csv_file) as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)    # skip header row

        for row in csv_reader:
            dataset_name = row[0] + '.csv'
            x_col = row[1]
            y_col = row[2]
            i_col = row[3]
            outlier = bool(row[4])

            ml.perform_lin_reg('data/' + dataset_name, x_col, y_col, i_col, outlier)

def main():
    run('datasets-regression.csv')
    
    
if __name__ == '__main__':
    main()





