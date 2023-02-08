from cal_ig import get_ig
# year = '2020'
# year = '2021'
year = '2022'
tf_idf_num = 1000
ig_word_num = 1000
if __name__ == '__main__':
    get_ig(year,tf_idf_num,ig_word_num)