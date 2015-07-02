'''
Created on 27 Mar 2015

@author: sduprey
'''

def save_histogram_as_csv_file(current_parameter, magasin, my_data,results_saving_path):
    file_path=unicode(results_saving_path+current_parameter+'_'+magasin+'.csv','utf-8')
    f=open(file_path, 'w')
    f.write(current_parameter+';'+'nb_occurences;\n')
    for row in my_data:
        for column in row:
                f.write(str(column)+';')
        f.write('\n')  
    f.close()
    print current_parameter;
    print magasin;
    print my_data;


def save_histogram_to_existing_csv_file(current_file, current_parameter, magasin, my_data):
    header_line=current_parameter+'_'+magasin+'\n'
    current_file.write(header_line)
    current_file.write(current_parameter+';'+'nb_occurences;\n')
    for row in my_data:
        for column in row:
                current_file.write(str(column)+';')
        current_file.write('\n')  
    print current_parameter;
    print magasin;
    print my_data;