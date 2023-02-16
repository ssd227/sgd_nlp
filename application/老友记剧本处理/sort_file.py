
def sort_file_process():
    with open('data/common_word.txt', 'r', encoding='utf-8') as f:
        x = [line.strip().split() for line in f.readlines() if len(line.strip().split())==2]
        


        x.sort(key=lambda x:int(x[1].strip()), reverse=True)

        x = [" ".join(line)+"\n" for line in x]

    with open('data/common_word.txt', 'w', encoding='utf-8') as f:
        f.writelines(x)
        
    

if __name__ == '__main__':
    sort_file_process()
    