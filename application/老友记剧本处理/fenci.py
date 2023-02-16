import os
import re

def load_common_word_dict():
    common_wd_dic = dict()
    with open('data/common_word.txt', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip().split()
            if len(word) == 2:
                common_wd_dic[word[0]] = 0
    return common_wd_dic

'''
    rule to remove common word
'''
def check_item_rules(item, common_word_dict):

    #rule 
    if len(item) > 0:
        if item[-1] in common_word_dict:
            item = item[:-1]

    # rule
    if len(item) > 0:
        if item[0] in common_word_dict:
            item = item[1:]

    return item

def process_all_files(file_dir_path):
    word_count  = dict()
    common_word_dict = load_common_word_dict()

    for home, dirs, files in os.walk(file_dir_path):
        for filename in files:
            fullname = os.path.join(home, filename)
            print('processing ', fullname)
            word_count = count_one_file_data(fullname, common_word_dict=common_word_dict, word_count=word_count)

    return word_count

def count_one_file_data(file_path, word_count, common_word_dict):
    with open(file_path, encoding='utf-8') as file:
        for line in file.readlines():
            
            ss = re.sub(r"[!:.?,()\-\[\]\']", " ", line)
            if len(ss) > 1:
                for ori_item in ss.split():
                    item = ori_item.strip()#.lower()
                    # check item
                    # item = check_item_rules(item, common_word_dict)
                    
                    
                    if item in common_word_dict or len(item) <=2:
                        continue
                    else:
                        if item not in word_count:
                            word_count[item] = 1
                        else:
                            word_count[item] += 1
    return word_count


def fenci():
    word_count = process_all_files('../../data/friends/season01/')


    res = [(k,v) for k, v in word_count.items() if v > 1]
    res.sort(key=lambda x:x[1], reverse=True)


    res_str = [x[0]+" "+str(x[1])+"\n" for x in res]
    with open('output/out_with_count.txt', 'w', encoding='utf-8') as f:
        f.writelines(res_str)
    
    # res_str = [x[0]+"\n" for x in res]
    # with open('output/out.txt', 'w', encoding='utf-8') as f:
    #     f.writelines(res_str)


if __name__ == '__main__':
    fenci()
    
