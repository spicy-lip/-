# -*- coding: utf-8 -*-
"""
电动车新国标评论分析 - 智能加权词云生成器（整合版）
功能：从原始评论CSV文件生成高质量、反映核心观点与公众影响力的词云
"""

import re
import jieba
import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 配置区 (用户可修改) ====================
class Config:
    # 数据文件配置
    INPUT_FILE = 'bilibili_新国标_汇总_comments_20251231_124435.csv'  # 输入文件，需包含'comment_text'和'like_count'列

    OUTPUT_PREFIX = 'new_standard'         # 输出文件前缀
    
    # 词云视觉配置
    FONT_PATH = 'C:/Windows/Fonts/msyh.ttc'             # 中文字体路径，务必确保存在
    WORDCLOUD_SIZE = (1200, 800)          # 词云图尺寸 (宽, 高)
    BACKGROUND_COLOR = 'white'            # 背景颜色
    MAX_WORDS = 250                       # 词云最多显示词数
    PREFER_HORIZONTAL = 0.7               # 水平排列偏好 (0-1)
    
    # 算法参数配置
    TFIDF_TOP_N = 200                     # TF-IDF保留的关键词数量
    LDA_TOPICS = 5                        # LDA主题数量
    LDA_WORDS_PER_TOPIC = 10              # 每个主题提取的关键词数
    MIN_WORD_LENGTH = 2                   # 词语最小长度
        # ====== 新增：情感分析配置 ======
    ENABLE_SENTIMENT_ANALYSIS = True    # 设为 True 以启用情感分析功能
    # 时间列名（根据你的数据文件中的实际列名修改，可选 ‘comment_time’， ‘created_at’， ‘time’等）
    TIME_COLUMN = 'comment_time'

# ==================== 2. 文本预处理与领域优化类 ====================
class CommentProcessor:
    """负责文本清洗、分词和领域优化"""
    
    def __init__(self):
        # 初始化正则表达式模式
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\S+')
        self.emoji_pattern = re.compile(r'\[.*?\]|【.*?】')
        self.noise_pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：,.!?;:"\'、（）《》【】#]')
        
        # 构建领域词典和停用词表
        self._build_domain_dictionary()
        self._build_stopwords()
        
    def _build_domain_dictionary(self):
        """针对电动车新国标领域添加专业术语"""
        domain_terms = [
            # 政策与标准相关
            '新国标', '旧国标', 'GB17761', '国家标准', '电动车', '电动自行车', 
            '电瓶车', '电摩', '超标车', '合规车', '过渡期',
            # 性能参数相关
            '25码', '25公里', '限速', '脚踏', '脚蹬子', '强制脚踏', '整车重量',
            '蓄电池', '电压', '电机功率', '续航里程', '最高车速',
            # 管理与使用相关
            '上牌', '登记', '牌照', '驾驶证', '非机动车道', '机动车道',
            '载人', '戴头盔', '安全帽', '闯红灯', '逆行', '酒驾',
            # 安全相关
            '锂电池', '铅酸电池', '充电安全', '自然', '起火', '爆炸',
            '防火阻燃', '短路保护', '过充保护',
            # 评价与态度相关
            '一刀切', '不合理', '懒政', '专家', '拍脑袋', '民意',
            '支持', '反对', '吐槽', '抱怨', '理解', '无奈'
        ]
        
        for term in domain_terms:
            jieba.add_word(term, freq=1000)
            # 同时添加无空格版本（防止误切）
            if ' ' in term:
                jieba.add_word(term.replace(' ', ''), freq=1000)
                
    def _build_stopwords(self):
        """构建通用+领域两级停用词表"""
        # 通用停用词
        self.general_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '个', '也', '很', '到', '说', '要', '去', '你',
            '会', '着', '没有', '看', '好', '自己', '这', '那', '啊',
            '哦', '嗯', '呃', '嘛', '呗', '啦'
        }
        
        # 领域无关停用词（平台特性、语气词）
        self.domain_stopwords = {
            '视频', 'up', 'up主', '博主', '作者', '弹幕', '评论',
            '哈哈', '哈哈哈', '呵呵', '嘻嘻', '嘿嘿', '啊啊', '哇塞',
            '请问', '谢谢', '感谢', '楼主', '沙发', '板凳', '前排',
            '我觉得', '我个人', '感觉', '认为', '以为', '可能', '也许',
            '说实话', '说真的', '坦白说', '其实', '然后', '那么', '所以'
        }
        
        self.all_stopwords = self.general_stopwords.union(self.domain_stopwords)
        
    def clean_text(self, text):
        """深度清洗单条文本"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
            
        text = str(text)
        # 顺序应用清洗规则
        text = self.url_pattern.sub(' ', text)
        text = self.mention_pattern.sub(' ', text)
        text = self.emoji_pattern.sub(' ', text)
        text = self.noise_pattern.sub(' ', text)
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def segment_text(self, text):
        """分词并过滤"""
        if not text:
            return []
            
        # 精确模式分词
        words = jieba.lcut(text, cut_all=False)
        # 过滤：长度要求 + 不在停用词表中
        filtered = [
            w for w in words 
            if len(w) >= Config.MIN_WORD_LENGTH 
            and w not in self.all_stopwords
        ]
        return filtered

# ==================== 3. 高级关键词筛选类 ====================
class KeywordSelector:
    """使用TF-IDF和LDA进行智能关键词筛选"""
    
    def __init__(self, documents):
        """
        documents: 列表，每条是已分词的评论（空格连接的字符串）
        """
        self.documents = documents
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def filter_by_tfidf(self, top_n=None):
        """使用TF-IDF提取重要关键词"""
        if top_n is None:
            top_n = Config.TFIDF_TOP_N
            
        if not self.documents:
            return set()
            
        # 将文档列表转换为TF-IDF矩阵
        tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 计算每个词的全局TF-IDF分数
        word_scores = tfidf_matrix.sum(axis=0).A1
        word_score_pairs = list(zip(feature_names, word_scores))
        
        # 按分数降序排序
        sorted_pairs = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)
        
        # 输出TF-IDF高分词（用于调试）
        print("\n" + "="*50)
        print("TF-IDF Top 30 关键词:")
        print("="*50)
        for word, score in sorted_pairs[:30]:
            print(f"{word}: {score:.4f}")
            
        # 返回前top_n个词的集合
        top_words = {word for word, _ in sorted_pairs[:top_n]}
        return top_words
        
    def identify_topics(self, n_topics=None, n_words_per_topic=None):
        """使用LDA发现核心讨论主题"""
        if n_topics is None:
            n_topics = Config.LDA_TOPICS
        if n_words_per_topic is None:
            n_words_per_topic = Config.LDA_WORDS_PER_TOPIC
            
        if len(self.documents) < n_topics:
            print("文档数量不足，跳过LDA分析")
            return []
            
        # 创建文档-词矩阵
        dtm = self.vectorizer.fit_transform(self.documents)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 训练LDA模型
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=50,
            learning_method='online'
        )
        lda.fit(dtm)
        
        # 提取每个主题的关键词
        topics = []
        print("\n" + "="*50)
        print("LDA主题分析结果:")
        print("="*50)
        
        for topic_idx, topic_weights in enumerate(lda.components_):
            top_indices = topic_weights.argsort()[-n_words_per_topic:][::-1]
            topic_keywords = [feature_names[i] for i in top_indices]
            topics.append(topic_keywords)
            
            print(f"\n主题 #{topic_idx + 1}:")
            print(", ".join(topic_keywords))
            
        return topics

# ==================== 4. 加权词频计算函数 ====================
def calculate_weighted_frequency(df, use_likes_weight=True, use_tfidf_filter=True, 
                                 tfidf_words=None, lda_topics=None):
    """
    计算加权词频
    df: 包含'segmented_words'和'like_count'的DataFrame
    """
    word_weights = defaultdict(float)
    
    for idx, row in df.iterrows():
        words = row['segmented_words']
        if not words:
            continue
            
        # 计算评论权重（基于点赞数）
        if use_likes_weight:
            likes = row.get('like_count', 0)
            # 使用对数平滑，平衡影响力
            comment_weight = math.log10(likes + 1) + 1
        else:
            comment_weight = 1.0
            
        # 对评论中的每个词加权累加
        for word in words:
            # 可选：使用TF-IDF筛选词
            if use_tfidf_filter and tfidf_words and word not in tfidf_words:
                continue
                
            word_weights[word] += comment_weight
            
    # 归一化处理（可选）
    if word_weights:
        max_weight = max(word_weights.values())
        if max_weight > 0:
            for word in word_weights:
                word_weights[word] = word_weights[word] / max_weight * 100
                
    return dict(word_weights)

# ==================== 5. 词云生成与可视化 ====================
def generate_wordcloud(weighted_freq_dict, output_filename=None):
    """生成并保存词云图"""
    if not weighted_freq_dict:
        print("错误：词频字典为空，无法生成词云")
        return None
        
    if output_filename is None:
        output_filename = f"{Config.OUTPUT_PREFIX}_wordcloud.png"
    
    # 配置词云参数
    wc = WordCloud(
        font_path=Config.FONT_PATH,
        width=Config.WORDCLOUD_SIZE[0],
        height=Config.WORDCLOUD_SIZE[1],
        background_color=Config.BACKGROUND_COLOR,
        max_words=Config.MAX_WORDS,
        collocations=False,  # 不计算搭配词
        prefer_horizontal=Config.PREFER_HORIZONTAL,
        contour_width=0,
        scale=2,
        max_font_size=200,
        min_font_size=10,
        random_state=42
    )
    
    # 生成词云
    print("\n正在生成词云...")
    wc.generate_from_frequencies(weighted_freq_dict)
    
    # 可视化设置
    plt.figure(figsize=(15, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('电动车新国标网络评论分析 - 加权词云', fontsize=20, pad=20, fontweight='bold')
    
    # 添加统计数据说明
    stats_text = f"总词汇数: {len(weighted_freq_dict)} | 最高频词: {max(weighted_freq_dict, key=weighted_freq_dict.get)}"
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12, style='italic')
    
    # 保存和显示
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=Config.BACKGROUND_COLOR)
    plt.show()
    
    print(f"✓ 词云图已保存至: {output_filename}")
    print(f"✓ 共可视化 {len(weighted_freq_dict)} 个关键词")
    
    return output_filename

# ==================== 6. 辅助分析函数 ====================
def analyze_top_keywords(weighted_freq_dict, top_n=20):
    """分析并展示Top关键词"""
    if not weighted_freq_dict:
        return
        
    sorted_keywords = sorted(weighted_freq_dict.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
    
    print("\n" + "="*50)
    print(f"Top {top_n} 关键词（加权后）:")
    print("="*50)
    
    for i, (word, weight) in enumerate(sorted_keywords[:top_n], 1):
        print(f"{i:2d}. {word:<10} : {weight:.2f}")
        
def save_keyword_data(weighted_freq_dict, df, filename_prefix=None):
    """保存关键词数据和中间结果"""
    if filename_prefix is None:
        filename_prefix = Config.OUTPUT_PREFIX
        
    # 保存加权词频
    freq_df = pd.DataFrame(
        weighted_freq_dict.items(), 
        columns=['keyword', 'weighted_frequency']
    ).sort_values('weighted_frequency', ascending=False)
    
    freq_df.to_csv(f"{filename_prefix}_keyword_frequencies.csv", 
                  index=False, encoding='utf-8-sig')
    print(f"✓ 关键词频率数据已保存至: {filename_prefix}_keyword_frequencies.csv")
    
    # 保存清洗后的评论
    if 'cleaned_text' in df.columns:
        df[['cleaned_text', 'like_count']].to_csv(
            f"{filename_prefix}_cleaned_comments.csv", 
            index=False, encoding='utf-8-sig'
        )
        print(f"✓ 清洗后评论数据已保存至: {filename_prefix}_cleaned_comments.csv")

# ==================== 7. 主执行函数 ====================
def main():
    """主函数：执行完整流程"""
    print("="*60)
    print("电动车新国标评论分析 - 智能词云生成系统")
    print("="*60)
    
    # 步骤1: 加载数据
    print("\n[步骤1/5] 正在加载数据...")
    try:
        df = pd.read_csv(Config.INPUT_FILE)
        print(f"✓ 成功加载数据，共 {len(df)} 条评论")
        
        # 检查必要列
        required_cols = ['comment_text']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"数据文件缺少必要列: {col}")
                
        # 确保点赞数列存在，如不存在则创建
        if 'like_count' not in df.columns:
            df['like_count'] = 0
            print("⚠ 未找到'like_count'列，已创建并设为0")
            
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return
        
    # 步骤2: 文本预处理
    print("\n[步骤2/5] 正在预处理文本...")
    processor = CommentProcessor()
    
    # 清洗文本
    df['cleaned_text'] = df['comment_text'].apply(processor.clean_text)
    
    # 分词
    df['segmented_words'] = df['cleaned_text'].apply(processor.segment_text)
    
    # 统计预处理结果
    original_word_count = sum(df['comment_text'].str.len())
    cleaned_word_count = sum(df['cleaned_text'].str.len())
    print(f"✓ 文本清洗完成: 字符数 {original_word_count} → {cleaned_word_count}")
    print(f"✓ 平均每条评论分词数: {df['segmented_words'].apply(len).mean():.1f}")
    
    # 步骤3: 高级关键词筛选
    print("\n[步骤3/5] 正在进行高级关键词筛选...")
    
    # 准备文档列表（用于TF-IDF和LDA）
    documents = [' '.join(words) for words in df['segmented_words'] if words]
    
    if documents:
        selector = KeywordSelector(documents)
        
        # TF-IDF筛选
        tfidf_keywords = selector.filter_by_tfidf()
        print(f"✓ TF-IDF筛选出 {len(tfidf_keywords)} 个重要关键词")
        
        # LDA主题分析
        lda_topics = selector.identify_topics()
        lda_keywords = set([word for topic in lda_topics for word in topic])
        print(f"✓ LDA分析发现 {len(lda_topics)} 个核心主题")
        
        # 合并关键词
        important_keywords = tfidf_keywords.union(lda_keywords)
        print(f"✓ 合并后重要关键词总数: {len(important_keywords)}")
    else:
        print("⚠ 无有效文档，跳过高级筛选")
        important_keywords = None
        lda_topics = []
    
    # 步骤4: 计算加权词频
    print("\n[步骤4/5] 正在计算加权词频...")
    
    # 这里可以修改参数以调整算法行为
    weighted_freq = calculate_weighted_frequency(
        df, 
        use_likes_weight=True,           # 启用点赞权重
        use_tfidf_filter=False,          # 是否使用TF-IDF筛选（True/False）
        tfidf_words=important_keywords,  # 筛选词集
        lda_topics=lda_topics            # LDA主题
    )
    
    print(f"✓ 词频计算完成，共得到 {len(weighted_freq)} 个有效词汇")
    
    # 分析Top关键词
    analyze_top_keywords(weighted_freq, top_n=20)
    
    # 步骤5: 生成词云并保存结果
    print("\n[步骤5/5] 正在生成词云和保存数据...")
    
    # 生成词云
    wordcloud_file = generate_wordcloud(weighted_freq)
    
    # 保存数据文件
    save_keyword_data(weighted_freq, df)
    
    # 最终统计报告
    print("\n" + "="*60)
    print("分析完成！总结报告:")
    print("="*60)
    print(f"• 处理评论总数: {len(df):,} 条")
    print(f"• 有效关键词数量: {len(weighted_freq):,} 个")
    print(f"• 最高权重关键词: {max(weighted_freq, key=weighted_freq.get)}")
    print(f"• 生成词云文件: {wordcloud_file}")
    print(f"• LDA发现主题数: {len(lda_topics)} 个")
        # 步骤5: 生成词云并保存结果 (原有代码不变)
    # ... [你原有的步骤5代码] ...

    # ====== 新增步骤6：情感趋势分析 ======
    if Config.ENABLE_SENTIMENT_ANALYSIS:
        try:
            # 调用情感分析函数，传入处理好的 DataFrame
            print("\n" + "="*60)
            print("启动情感趋势分析模块...")
            print("="*60)
            quick_sentiment_analysis(df)
        except Exception as e:
            print(f"情感分析模块执行失败，但不影响主流程。错误信息: {e}")
    # ====== 新增代码结束 ======

    # 最终统计报告 (原有的报告，保持不变)
    print("\n" + "="*60)
    print("分析完成！总结报告:")
    # ... [你原有的报告打印代码] ...
    print("="*60)

# ==================== 新增：极简情感趋势分析函数 ====================
from snownlp import SnowNLP

def quick_sentiment_analysis(df):
    """
    快速情感分析并生成图表
    df: 主流程中处理好的 DataFrame，应包含 ‘cleaned_text‘, ‘like_count‘， 并尝试寻找时间列
    """
    import matplotlib
    try:
        matplotlib.font_manager.fontManager.addfont(Config.FONT_PATH)
        font_name = matplotlib.font_manager.FontProperties(fname=Config.FONT_PATH).get_name()
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        print(f"✓ 已加载指定字体: {font_name}")
    except Exception as e:
        print(f"⚠ 加载指定字体失败 ({Config.FONT_PATH})，尝试系统字体。错误: {e}")
        # 方案2：备用方案，尝试常用系统字体
        fallback_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
        matplotlib.rcParams['font.sans-serif'] = fallback_fonts
    # 确保能正常显示负号
    matplotlib.rcParams['axes.unicode_minus'] = False
    # ========== 字体设置结束 ==========
    print("\n[快速分析] 正在生成情感趋势图表...")
    # 1. 情感打分与分类 (核心代码)
    def get_sentiment(text):
        try:
            # SnowNLP情感值范围0-1，>0.6视为正面，<0.4视为负面，中间为中性
            return SnowNLP(str(text)).sentiments
        except:
            return 0.5  # 解析失败视为中性
    
    df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment)
    df['sentiment'] = df['sentiment_score'].apply(
        lambda x: '正面' if x > 0.6 else ('负面' if x < 0.4 else '中性')
    )
    
    # 2. 计算加权情感（考虑点赞数）
    df['weighted_sentiment'] = df['sentiment_score'] * (df['like_count'] + 1)
    
    # 3. 准备时间序列（如果存在时间列）
    time_column_name = None
    for possible_col in [Config.TIME_COLUMN, 'created_at', 'time', '评论时间']:
        if possible_col in df.columns:
            time_column_name = possible_col
            print(f"✓ 找到时间列: '{time_column_name}'")
            break
    
    daily_trend = None
    if time_column_name:
        try:
            df['date'] = pd.to_datetime(df[time_column_name]).dt.date
            # 按日期和情感分组，计算每日评论数量
            daily_trend = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        except Exception as e:
            print(f"⚠ 时间列处理失败，将仅输出总体分布图。错误: {e}")
            daily_trend = None
    else:
        print("⚠ 未找到有效的时间列，将仅输出总体分布图。")
    
    # 4. 绘制图表

    # ---------- 修复关键：先计算饼图所需数据，与时间趋势无关 ----------
    # 计算整体情感分布 (饼图核心数据)
    sentiment_counts = df['sentiment'].value_counts()
    # 定义统一的颜色映射
    colors_map = {'正面': '#4CAF50', '中性': '#2196F3', '负面': '#FF5252'}
    colors = [colors_map.get(x, 'grey') for x in sentiment_counts.index]
    # -----------------------------------------------------------------

    # 根据是否有时间数据决定画布布局
    if daily_trend is not None and len(daily_trend) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]  # 便于统一处理

    # 图表1：整体情感分布饼图 (现在sentiment_counts已定义)
    wedges, texts, autotexts = axes[0].pie(
        sentiment_counts.values,
        labels=None,  # 用图例代替标签
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.85,
        textprops={'fontsize': 11}
    )
    axes[0].set_title('整体情感分布 (基于评论条数)', fontsize=15, fontweight='bold', pad=20)
    axes[0].legend(
        wedges,
        sentiment_counts.index,
        title="情感倾向",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=11
    )

    # 图表2（如果时间数据有效）：每日情感趋势堆叠面积图
    if daily_trend is not None and len(daily_trend) > 1:
        daily_trend = daily_trend.reindex(columns=['正面', '中性', '负面'], fill_value=0)
        daily_trend.plot(
            kind='area',
            ax=axes[1],
            stacked=True,
            color=colors_map,  # 使用前面已定义的colors_map
            alpha=0.7,
            linewidth=0.5
        )
        axes[1].set_title('每日情感趋势演化', fontsize=15, fontweight='bold', pad=20)
        axes[1].set_xlabel('日期', fontsize=12, labelpad=10)
        axes[1].set_ylabel('评论数量 (条)', fontsize=12, labelpad=10)
        axes[1].xaxis.set_tick_params(rotation=30)
        axes[1].legend(title="情感倾向", title_fontsize=11, fontsize=11, loc='upper left')
        axes[1].grid(True, alpha=0.3, linestyle='--', which='major')

    plt.tight_layout()

    # 5. 保存图表
    output_file = f"{Config.OUTPUT_PREFIX}_sentiment_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✓ 情感分析图表已保存至: {output_file}")

    # 6. 打印关键统计数据（可直接写入论文）
    total = len(df)
    print(f"\n{'='*50}")
    print("关键统计结果 (可直接用于论文):")
    print("="*50)
    print("1. 总体情感分布:")
    for label, count in sentiment_counts.items():
        print(f"   - {label}: {count} 条 ({count/total*100:.1f}%)")

    print(f"\n2. 加权后平均情感得分: {df['weighted_sentiment'].mean():.3f}")
    print("   (分数解释: 越接近1表示舆论越正面，越接近0表示越负面)")

    if daily_trend is not None:
        print(f"\n3. 分析时间段: {df['date'].min()} 至 {df['date'].max()}")
        print(f"   共覆盖 {len(daily_trend)} 天")
    print("="*50)
    
    # 可选：保存带有情感标签的数据，供深度分析
    sentiment_output_file = f"{Config.OUTPUT_PREFIX}_comments_with_sentiment.csv"
    df[['cleaned_text', 'sentiment', 'sentiment_score', 'like_count', 'date' if 'date' in df.columns else 'date']].to_csv(
        sentiment_output_file, index=False, encoding='utf-8-sig'
    )
    print(f"✓ 带情感标签的数据已保存至: {sentiment_output_file}")
    
    return fig

# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 执行主函数
    main()