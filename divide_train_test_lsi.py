import random

train_test_split = [0.9,0.0,0.1]
with open("datadump/train.csv",'rb') as file_in,  open('datadump/lsi_data/train.csv','wb') as train_out,open('datadump/lsi_data/valid.csv','wb') as valid_out,open('datadump/lsi_data/test.csv','wb') as test_out:
    first_line = True
    for line in file_in:
        if first_line == True:
            train_out.write(line +"\n")
            valid_out.write(line +"\n")
            test_out.write(line +"\n")
            first_line = False 
        rand_num = random.random()
#         print rand_num
        if rand_num<train_test_split[0]:
            train_out.write(line +"\n")
        elif rand_num<train_test_split[0] + train_test_split[1]:
            valid_out.write(line +"\n")
        else:
            test_out.write(line +"\n")
        
             