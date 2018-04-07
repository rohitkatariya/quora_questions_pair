import csv
import sys
sys.path.append("/data/analytics/rohit/sentence_classification/python_files/libraries/")
from stringFunctions import clean_text
from dictFunctions import clean_dict_strings, clean_dict_list_strings
# readCsvToDictList , readCsvToList , readAtomicCsvToList , readCsvAsKeyValue
# writeTuplesListToCsv , writeDictListToCSV , writeDictToCSV , writeAtomicListToCSV , 
def readCsvToDictList(filename, delimiter=None, quotechar="\""):
    """
        reads csv file and convert each row to a dict with keys as first row.
        CAUTION: the first row should contain the column names 
    """
    
    rows_to_return = []
    with open(filename, 'rU') as csvfile:
        
        if delimiter:
            reader = csv.DictReader(csvfile, delimiter=delimiter, quotechar=quotechar)
        else:
            reader = csv.DictReader(csvfile, quotechar=quotechar)
        
        for row in reader:
            rows_to_return.append(row)
#             print row
    return rows_to_return
            
    
def readCsvToList(filename):
    rls = []
    with open(filename, 'rU') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            rls.append(row)
    return rls

def readAtomicCsvToList(filename):
    rls = []
    with open(filename, 'rU') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            rls.append(row[0])
    return rls
def writeTuplesListToCsv(data, outputFileName, headers=None, file_mode='w'):
    """
        Outputs a list of tuples to a csv file
    """
    with open(outputFileName + '.csv', file_mode) as out:
        csv_out = csv.writer(out, lineterminator="\n")
        if headers:
            csv_out.writerow(headers)
        for row in data:
            csv_out.writerow(row)
            

def writeDictListToCSV(dicList, filename, delimiter=None, filemode='wb', column_names_set=None):
    """
        Writes a list of dictionaries to a csv file with each dictionary as a row
    """   
    if not column_names_set:
        column_names_set = set([])
        for row in dicList:
            for k in row.keys():
                column_names_set.add(k)
    
    with open(filename + '.csv', filemode) as output_file:
        
        if not delimiter:
            dict_writer = csv.DictWriter(output_file, column_names_set)
        else:
            dict_writer = csv.DictWriter(output_file, column_names_set, delimiter=delimiter)
        dict_writer.writeheader()
        dict_writer.writerows(clean_dict_list_strings(dicList))
#     print column_names_set    

def writeDictToCSV(dict_, filename):
    """
         Writes dictionary to csv file with each [key ,value] pair as a row
    """
    with open(filename + '.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_.items():
           writer.writerow([key, value])
           
def writeAtomicListToCSV(list_, filename, headers=None):
    """
         Writes list of single values to csv file with each [value] as a row
    """
    with open(filename + '.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        if headers:
            writer.writerow(headers)
        for  value in list_:
           writer.writerow([ value])

def readCsvAsKeyValue(filename, delimiter=None, skip_header=False, lower_strip_keys=False):
    dict_to_return = {}
    with open(filename) as csvfile:
        if delimiter:
            reader = csv.reader(csvfile, delimiter=delimiter)
        else:
            reader = csv.reader(csvfile)
        if skip_header:
            reader.next()    
        for row in reader:
            if lower_strip_keys:
                row[0] = row[0].lower().strip()
            dict_to_return[row[0]] = row[1] 
    return dict_to_return

if __name__ == "__main__":
    print(readCsvToList("/home/kr/Desktop/naukri_python_files/crawl_for_linkedin_skills/AutoSuggest_Libraries.csv")[1])

