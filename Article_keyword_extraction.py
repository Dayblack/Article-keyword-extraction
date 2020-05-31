#!/usr/bin/env python
# coding: utf-8

# In[25]:


import jieba.posseg as pseg
import os
class CandidateWords:
    def __init__(self):
        self.stopws = [] #创建一个停用词列表
        self.candidate_word = [] #创建一个候选词列表
        self.flag = [] #创建一个候选词词性列表
        self.candidate_dict = {} #创建一个{candidate_word:flag}字典
        self.nword = {} #candidate_word 元组（去重）
        
    def get_stopwd(self):
        """
        function:获取停用词表（停用词来源：哈工大+百度停用词表）
        return:停用词列表self.stopws
        """
        
        base_dir = os.path.dirname(os.path.abspath("__file__")) #获取到当前文件夹的一个绝对路径
        file_path = os.path.join(base_dir,"stopwords.txt") #获取停用词文件的完整路径
        files = open(file_path,"r",encoding="utf-8") #打开文件，形成类文件对象
        stop_words = files.readlines()
        for line in stop_words:
            sw = line.strip()

            self.stopws.append(sw)
        files.close()
        return self.stopws #得到停用词列表
    
    def get_candidate_list(self,string_sentence):
        """
        function:使用停用词进行过滤
        string_sentence:要分析的语句
        return:经过停用词过滤之后的候选词和它的词性candidate_dict；候选词词典及其初始化的权重nword
        """
        stop_words = self.get_stopwd()#得到停用词列表
        words_tag = pseg.cut(string_sentence)
        for w in words_tag:
            if w.flag != u"x" and (w.word not in stop_words):
                self.candidate_word.append(w.word.encode("utf-8")) # 去除停用词后的候选词candidate_word
                self.flag.append(w.flag.encode("utf-8")) # 保留候选词的词性
        for i in range(len(self.flag)):
            self.candidate_dict[self.candidate_word[i]] = self.flag[i] #建立别划分词的字典，键为该词，值为该词对应的词性
        for i in range(len(self.candidate_word)):
            self.nword[i] = self.candidate_word[i]
        return self.candidate_dict,self.nword#字典
    

import numpy
from igraph import *
class SemanticSimilarity:
    def __init__(self):
        self.word_tag_dict = {}
        self.E = []
    def word_tag_dictionary(self):
        """
        function:获取同义词林
        return:返回同义词林self.word_tag_dict
        """
        
        base_dir = os.path.dirname(os.path.abspath("__file__"))
        file_path = os.path.join(base_dir,"word_codes.txt")#得到word_codes的完整文件地址
        files = open(file_path,"r",encoding="utf-8")
        table = files.readlines()
        for code in table:
            code = code.strip()
            codes = code.split(" ")
            self.word_tag_dict[codes[0]] = codes[1:]
        files.close()
        return self.word_tag_dict#以字典的形式返回key同义词林每个词和value对应的编码
    @staticmethod
    def similarity(i, j, candidate_word, word_tag_dict):
        """
        function:　计算编码之间的距离
        :param i: 候选词ｉ
        :param j: 候选就j
        :param candidate_word: 候选词表
        :param word_tag_dict: 同义词林
        :return: sim: 相似度
        """
        weights_set = [1.0, 0.5, 0.25, 0.25, 0.125, 0.06, 0.06, 0.03]  # 不同等级的编码距离的权重
        alpha = 5
        init_dis = 10
        temp_list = list()
        w1 = candidate_word[i]#从候选词表中取出候选词
        w2 = candidate_word[j]
        code1 = word_tag_dict[w1]#根据上面取出的候选词，取出对应的编码
        code2 = word_tag_dict[w2]
        for m in range(len(code1)):#根据编码长度
            for n in range(len(code2)):
                diff = -1
                for k in range(len(code2[n])):
                    if code1[m][k] != code2[n][k]:    # compare code
                        diff = k

                        temp_list.append(diff)
                        break
                if (diff == -1) and (code2[n][7] != u'#'):
                    sim = 1.0
                    return sim
                elif (diff == -1) and (code2[n][7] == u'#'):
                    min_dis = weights_set[7]*init_dis
                    sim = alpha / (min_dis+alpha)
                    return sim
        diff = min(temp_list)
        min_dis = weights_set[diff]*init_dis
        sim = alpha / (min_dis+alpha)
        return sim#最后返回候选词i与候选词j之间的相似度
    def similar_matrix(self,string_data):
        """
        function:构建语义相关度网络
        string_data:待分析的语句
        return:语义相关度网络similar_matrix,二维对称矩阵
        """
        word_tag_dict = self.word_tag_dictionary()#词林字典
        keys = word_tag_dict.keys()#获取该词林的词列表
        #经过停用词过滤之后的候选词和它的词性candidate_dict；候选词词典及其初始化的权重nword
        candidate_words_dict, nwword = CandidateWords().get_candidate_list(string_data)
        nwword_words = list(nwword.values())#获取候选词列表
        length = len(nwword_words)
        similar_matrix = numpy.zeros(shape=(length, length))#建立方阵
        word_list = list()
        for word in nwword_words:
            if word in keys:
                word_list.append(word)#如果词在词林中有对应的编码，则将其加入到word_list列表中
        for i in range(length):
            for j in range(length):
                if (nwword_words[i] in word_list) and (nwword_words[j] in word_list):#遍历候选词列表
                    similar_matrix[i][j] = self.similarity(i, j, nwword_words, word_tag_dict)#确保在词林中出现的词可以计算相似度sim
                else:
                    similar_matrix[i][j] = 0.2#词林中没有的词默认为相似度sim为0.2
        return similar_matrix#词语词之间的相似矩阵构建完毕
    def similarity_network_edges(self, string_data):
        """
        对冗余的对称阵化简，取上三角阵，并将相似度大于0.5（阈值）的候选词索引（元组）加入到列表E中
        列表E中储存候选词中相似词的位置
        """
        similar_matrix = self.similar_matrix(string_data)
        row_col = similar_matrix.shape
        for i in range(row_col[0]):
            for j in range(i+1, row_col[0]):
                if similar_matrix[i][j] > 0.5:
                    self.E.append((i, j))
        return self.E
    @staticmethod
    def draw_network(matrix, label):
        """
        function: 画出词语的语义相关度网络
        :param matrix: 词语语义相关度矩阵
        :param label: 词语的标签名
        :return: none
        """
        g = Graph(matrix.__len__())
        g.vs["label"] = label
        edges = []
        weights = []
        for i in range(0, matrix.__len__()):
            for j in range(0, matrix.__len__()):
                if matrix[i][j] > 0.5:
                    edges += [(i, j)]
                    weights.append(matrix[i][j])
            g.add_edges(edges)
        g = g.simplify()
        layout =g.layout_graphopt()
        p = Plot()
        p.background = "#ffffff"     # 将背景改为白色，默认是灰色网格
        p.add(g,
        bbox=(50, 50, 550, 550),    # 设置图占窗体的大小，默认是(0,0,600,600)
        layout =layout,                        # 图的布局
        vertex_size=10,      # 点的尺寸
        edge_width=0.5, edge_color="grey", # 边的宽度和颜色，建议灰色，比较好看
        vertex_label_size=10,           # 点标签的大小
        vertex_color = "pink")  # 为每个点着色
        p.save("SNA.png")  # 将图保存到特定路径，igraph只支持png和pdf
        p.remove(g)         # 清除图像
import networkx as nx
class BetweenCentrality:

    def __init__(self):
        self.G = nx.Graph()
        self.bcdict = {}
        self.nword = {}

    def codes_betweeness_centarlity(self, string_sentence):
        """
        function: 计算词语的居间度
        :param: string_sentence :  待分析的短句
        :return: self.bcdict : 词语居间度 字典
        """
        candidate_words_dict, nwword = CandidateWords().get_candidate_list(string_sentence)
        nwword_words = list(nwword.values())
        length = len(nwword_words)
        for i in range(length):
            self.G.add_node(i)
        E = SemanticSimilarity().similarity_network_edges(string_sentence)#相似词索引（i，j）列表
        self.G.add_edges_from(E)
        vd = nx.betweenness_centrality(self.G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
        for i in range(length):
            self.bcdict[nwword_words[i]] = vd[i]
        for i in range(length):
            self.nword[i] = nwword_words[i]
        return self.bcdict#返回值为每个候选词的居间度 键为候选词，值为居间度

from collections import Counter

import os
import jieba.analyse
class Keyword:

    def __init__(self):
        self.poss = {}   # 词性表
        self.word_length = {}  # 词长表(记录每个词的词长)
        self.word_score = {}  # 词的评分表
    def feature(self, string_data):
        """
        function: 计算候选词的词性权重，词频，词长
        :param string_data: 待分析的语句
        :return:
        """
        base_dir = os.path.dirname(os.path.abspath("__file__"))
        file_path = os.path.join(base_dir,"tag.txt")
        files = open(file_path,"r",encoding="utf-8")
        poss_file = files.readlines()
        for line in poss_file:
            s = line.strip().split(' ')
            self.poss[s[0]] = s[1]
        po = self.poss#词性权重字典po创建完成
        candidate_words_dict, nword = CandidateWords().get_candidate_list(string_data)
        nwword_words = nword.values()
        pos = {}
        for word in nwword_words:
            self.word_length[word] = len(word)/3
            if candidate_words_dict[word] in po.keys():
                pos[word] = float(po[candidate_words_dict[word]])#如果候选词词性出现在了已有的词性字典中，则根据已有字典的该词权重
            else:
                pos[word] = 0.1#否则，默认权重为0.1
        #pos为候选词词性权重字典
        words_tf_dict = dict(Counter(nwword_words))#统计候选词中各词出现的次数，key为候选词，value为出现次数
        files.close()
        return pos, words_tf_dict, self.word_length, nwword_words#至此，得到了每一个候选词的居间度、出现词频、该候选词的词性权重、候选词的长度
    def score(self, string_data):
        """
        function: 计算候选词的重要性权重(weight)
        :param string_data: 待分析的短句
        :return: 候选词的权重排位
        """
        tw = 0.4  # 词权重
        vdw = 0.6  # 居间度权重
        lenw = 0.1  # 词长权重
        posw = 0.8  # 词性权重
        tfw = 0.3  # tf词频权重
        
        pos, words_tf_dict, word_length, candidate_word = self.feature(string_data)
        vd = BetweenCentrality().codes_betweeness_centarlity(string_data)
        for word in candidate_word:
            #每一个候选词的得分 s字典
            s = (vd[word] * vdw ) + (tw * (word_length[word] * lenw + pos[word] * posw + words_tf_dict[word]*tfw))
            self.word_score[word] = s
        rank = sorted(self.word_score.items(), key=lambda d: d[1], reverse=True)#由得分高到得分低依次排列
        return rank#返回列表[(候选词1，得分),(候选词2，得分),()...]
    def keyword(self, string_data):
        """
        function: 返回关键词及其评分
        :param string_data: 待分析的短句
        :return: keywords :关键词，关键词评分
        """
        key_score = self.score(string_data)
        keywords = []
        for key in key_score[0:20]:
            keywords.append(key[0])
        return keywords, key_score#返回得分最高的7个候选词keywords，与候选词列表[(候选词1，得分),(候选词2，得分),()...]
  
def main_keyword(news_list):
    string = ''
    for i in news_list:
        string += i
    keyword_list = Keyword().keyword(string)
    keywords = keyword_list[0]
    print('------------本程序的提取效果--------------')
    for key in keywords:
        key = str(key,encoding="utf-8")
        print(key)
    print('------------结巴分词的提取效果--------------')
    jieba_list = jieba.analyse.extract_tags(string)
    for key in jieba_list:
        print(key)
    print('\n')
    return keywords


if __name__ == "__main__":
    string = open('test.txt','r',encoding= "utf-8",errors="ignore").read().split('\n')
    keywords = main_keyword(string)
    keys={}
    score=100
    for key in keywords:
#         print(key)
        keys['key']=score
        score = score-10
#     print(keywords)


# In[ ]:





# In[ ]:




