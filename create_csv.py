import os 
import csv
idx = 0
person = 0
with open(R'./DataList/train.csv','w', newline = '') as train:
    with open(R'./DataList/val.csv','w', newline = '') as val:
        with open(R'./DataList/test.csv','w', newline = '') as test:
            csv_writer = csv.writer(train)
            csv_writer.writerow(['image','id'])
            csv_writer1 = csv.writer(val)
            csv_writer1.writerow(['image','id'])
            csv_writer2 = csv.writer(test)
            csv_writer2.writerow(['image','id'])
            for root, dirs, filenames in os.walk(r'./Database/cifar10'):
                for k in dirs: 
                    for root1, dirs1, filenames1 in os.walk(r'./Database/cifar10/{}'.format(k)):   
                        for i in filenames1:
                            idx += 1
                            data_path = os.path.join(root1,i)
                            print(idx)
                            if len(filenames1)*0 < idx <= len(filenames1)*0.8:
                                csv_writer.writerow([data_path,person])
                            if len(filenames1)*0.8 < idx <= len(filenames1)*0.9:
                                csv_writer1.writerow([data_path,person])
                            if len(filenames1)*0.9 < idx <= len(filenames1)*1.0:
                                csv_writer2.writerow([data_path,person])
                    idx = 0
                    person += 1