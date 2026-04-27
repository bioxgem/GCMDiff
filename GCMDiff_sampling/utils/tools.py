import os

def revearseimg(transposed_data):
    for i in range(transposed_data.shape[0]):
        for j in range(transposed_data.shape[1]):
            transposed_data[i][j][0]=(transposed_data[i][j][0])*20
            transposed_data[i][j][1]=(transposed_data[i][j][1])*20
            transposed_data[i][j][2]=((transposed_data[i][j][2]))*4
    return transposed_data

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  
        except Exception as e:
            print(f'無法刪除 {file_path}. 原因: {e}')