import sys
import pandas as pd

def load_excel_data(path):
    data_xls = pd.read_excel(path, 'Sheet1')
    return data_xls

def convert_feature(train_data, test_data):
    s = {}
    train_len = len(train_data)
    test_len = len(test_data)
    del train_data['学生ID']
    del test_data['学生ID']
    del train_data['裸眼视力(左)']
    del test_data['裸眼视力(左)']
    del train_data['裸眼视力(右)']
    del test_data['裸眼视力(右)']
    train_y = train_data['综合GPA']
    train_x = train_data.drop(['综合GPA'],axis=1)
    test_x = test_data
    for i in range(train_len):
        for col in train_x.loc[[i]]:
            try:
                train_x[col][i] = float(train_x[col][i])
            except:
                item = str(train_x[col][i])
                if col not in s.keys():
                    tmp = {}
                    tmp[item] = 1
                    s[col] = tmp
                elif item not in s[col].keys():
                    s[col][item] = 1
    Lsts1 = {}
    Lsts2 = {}
    for col in s:
        for col2 in s[col]:
            Lsts1[col+'is'+col2] = []
            Lsts2[col+'is'+col2] = []

    for i in range(train_len):
        print ('train'+str(i))
        for col in s:
            item = str(train_x[col][i])
            for col2 in s[col]:
                if col2 == item:
                    Lsts1[col+'is'+item].append(1)
                else:
                    Lsts1[col+'is'+col2].append(0)
    
    for i in range(test_len):
        print ('test'+str(i))
        for col in s:
            item = str(test_x[col][i])
            for col2 in s[col]:
                if col2 == item:
                    Lsts2[col+'is'+item].append(1)
                else:
                    Lsts2[col+'is'+col2].append(0)
    for col in s:
        del test_x[col]
        del train_x[col]
        for col2 in s[col]:
            train_x[col+'is'+col2] = Lsts1[col+'is'+col2]
            test_x[col+'is'+col2] = Lsts2[col+'is'+col2]
    # import pdb
    # pdb.set_trace()

    return train_x, train_y, test_x

if __name__ == '__main__':
    data1 = load_excel_data(sys.argv[1])
    data2 = load_excel_data(sys.argv[2])
    convert_feature(data1,data2)
