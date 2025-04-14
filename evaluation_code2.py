
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
from sklearn.model_selection import train_test_split
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
# import pyLDAvis
# import pyLDAvis.lda_model
from gensim.models import LdaModel
from multiprocessing import freeze_support
import numpy as np
# import IPython
# import pyLDAvis.gensim

# # Загрузка стоп-слов и лемматизатора
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)
# stop_words = stopwords.words('english')
# lemmatizer = WordNetLemmatizer()

# # Функция для очистки текста
# def clean_text(text):
#     text = re.sub('[^а-яёa-z ]', ' ', text, flags=re.IGNORECASE)
#     text = re.sub(' +', ' ', text)
#     text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
#     return text.lower()



# Загрузка необходимых ресурсов NLTK (если еще не загружены)
nltk.download('punkt_tab')
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)  # Необходимо для word_tokenize
#nltk.download('wordnet', quiet=True) # Stemming не требует wordnet
#nltk.download('omw-1.4', quiet=True) # Stemming не требует omw


stop_words_unfilled = set(stopwords.words('english')) # Преобразуем в set для скорости

# Расширенный список стоп-слов (добавьте свои)
extra_stop_words = {
    'suicide', 'drug', 'drugs', 'tramp', # Нежелательные слова (добавьте свои)
    'get', 'taken', 'take', # Английский глагол (если это проблема)
    'film', 'movie', # Слова, часто встречающиеся в описаниях фильмов
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', # Общие английские слова
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',  # числительные
    'ha', 'case', 'man', 'men', 'dude', 'thus', 'often', 'time', 'th', # слишком общие слова
    'corleone', 'michael', 'tony', 'peter', 'jesse', 'nick', 'crowe', 'vito', 'wallace', 'tommy', 'travis', 'jeff', 'riley', 'jones', 'anthony', 'jeanne', 'kane', 'lina', 'sophie', 'ford', 'cobb', 'john', 'jack', 'tony', # имена
}

stop_words = stop_words_unfilled.union(extra_stop_words)  # Объединяем списки

stemmer = PorterStemmer()

def clean_text(text):
    """
    Очищает текст, проводит токенизацию, удаляет стоп-слова и выполняет стемминг.

    Args:
      text: Исходный текст.

    Returns:
      Строка, содержащая обработанный текст.
    """

    # 1. Удаление не-буквенных символов и лишних пробелов
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Оставляем только буквы и пробелы
    text = re.sub(r'\s+', ' ', text).strip() # Убираем лишние пробелы и обрезаем по краям

    # 2. Токенизация
    tokens = word_tokenize(text)

    # 3. Удаление стоп-слов и приведение к нижнему регистру
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]

    # 4. Стемминг
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # 5. Объединение токенов обратно в строку
    cleaned_text = ' '.join(stemmed_tokens)

    return cleaned_text


# Функция для расчета когеренции
def compute_coherence_values(model, corpus, dictionary, texts):
    coherence_model = CoherenceModel(model=model, texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

if __name__ == '__main__':
    freeze_support()

    # Загрузка данных
    data = pd.read_csv('C:/Users/marib/Desktop/files/maga2/диссер/en_lsa_lda/preprocessed_dataset.csv', header=0)
    # data['cleaned'] = data['plot'].apply(clean_text)
    # df = data['cleaned']
    df = data['plot'].apply(clean_text)

    # Векторизация текста
    vect = TfidfVectorizer(stop_words=list(stop_words), max_features=1000)
    df = pd.DataFrame(df)
    vect_text = vect.fit_transform(df['plot'])
    idf = vect.idf_
    dd = dict(zip(vect.get_feature_names_out(), idf))
    l = sorted(dd, key=(dd).get)

    # Разделение данных на тренировочную и тестовую выборки
    X_train, X_test = train_test_split(vect_text, test_size=0.2, random_state=42)

    # LSA
    lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)
    lsa_top = lsa_model.fit_transform(vect_text)

    # LDA
    lda_model_sklearn = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=42, max_iter=1)
    lda_model_sklearn.fit(X_train)
    lda_top = lda_model_sklearn.transform(vect_text)

    # Создаем словарь и корпус gensim
    id2word = corpora.Dictionary([vect.get_feature_names_out()])
    corpus = [id2word.doc2bow(doc.split()) for doc in df['plot']]

    # Создадим папку для результатов, если ее нет
    output_dir = "output_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === Вывод для LSA ===
    with open(os.path.join(output_dir, "lsa_topics.txt"), "w", encoding="utf-8") as f:
        f.write("=== LSA ===\n")
        vocab = vect.get_feature_names_out()
        for i, comp in enumerate(lsa_model.components_):
            vocab_comp = zip(vocab, comp)
            sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
            f.write(f"Topic {i}: ")
            for t in sorted_words:
                f.write(t[0] + " ")
            f.write("\n")

            # Получаем фильмы, наиболее связанные с этой темой
            topic_docs = sorted(range(len(lsa_top)), key=lambda k: lsa_top[k][i], reverse=True)
            f.write("  Фильмы в этой теме:\n")
            for doc_id in topic_docs[:5]:  # Покажем топ-5 фильмов
                f.write(f"  - {data['название (ориг)'].iloc[doc_id]}\n")
            f.write("-" * 30 + "\n")

    # === Вывод для LDA ===
    with open(os.path.join(output_dir, "lda_topics.txt"), "w", encoding="utf-8") as f:
        f.write("=== LDA ===\n")
        vocab = vect.get_feature_names_out()
        for i, comp in enumerate(lda_model_sklearn.components_):
            vocab_comp = zip(vocab, comp)
            sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
            f.write(f"Topic {i}: ")
            for t in sorted_words:
                f.write(t[0] + " ")
            f.write("\n")

            # Получаем фильмы, наиболее связанные с этой темой
            topic_docs = sorted(range(len(lda_top)), key=lambda k: lda_top[k][i], reverse=True)
            f.write("  Фильмы в этой теме:\n")
            for doc_id in topic_docs[:5]:  # Покажем топ-5 фильмов
                f.write(f"  - {data['название (ориг)'].iloc[doc_id]}\n")
            f.write("-" * 30 + "\n")

    # === Вывод тем для каждого фильма (LSA) ===
    with open(os.path.join(output_dir, "lsa_movie_topics.txt"), "w", encoding="utf-8") as f:
        f.write("=== Темы для каждого фильма (LSA) ===\n")
        for doc_id in range(len(lsa_top)):
            f.write(f"Фильм: {data['название (ориг)'].iloc[doc_id]}\n")
            f.write("  Темы:\n")
            for topic_id in range(len(lsa_model.components_)):
                f.write(f"   - Topic {topic_id}: {lsa_top[doc_id][topic_id]:.3f}\n")
            f.write("-" * 30 + "\n")

    # === Вывод тем для каждого фильма (LDA) ===
    with open(os.path.join(output_dir, "lda_movie_topics.txt"), "w", encoding="utf-8") as f:
        f.write("=== Темы для каждого фильма (LDA) ===\n")
        for doc_id in range(len(lda_top)):
            f.write(f"Фильм: {data['название (ориг)'].iloc[doc_id]}\n")
            f.write("  Темы:\n")
            for topic_id in range(len(lda_model_sklearn.components_)):
                f.write(f"   - Topic {topic_id}: {lda_top[doc_id][topic_id]:.3f}\n")
            f.write("-" * 30 + "\n")

    print(f"Результаты сохранены в папку '{output_dir}'.")
    
    # === Оценка LDA ===
    # Перплексия
    perplexity = lda_model_sklearn.perplexity(X_test)
    print(f"Перплексия LDA: {perplexity}")


    # Преобразуем LDA модель sklearn в gensim LdaModel
    lda_model_gensim = LdaModel(corpus=corpus,
                            id2word=id2word,
                            num_topics=lda_model_sklearn.n_components,
                            random_state=42,
                            passes=10, # сколько раз можель будет проходить через корпус во время обучения
                            alpha='auto',
                            eta='auto')
    
    # Вычисляем когеренцию для LDA
    lda_coherence = compute_coherence_values(model=lda_model_gensim, corpus=corpus, dictionary=id2word, texts=df['plot'])
    print(f"Когеренция LDA: {lda_coherence}")

    # Для LSA вычисляем когеренцию на основе топа слов.
    lsa_topics_for_coherence = []
    for i, comp in enumerate(lsa_model.components_):
        vocab_comp = zip(vect.get_feature_names_out(), comp)
        sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
        lsa_topics_for_coherence.append([word for word, _ in sorted_words])
    lsa_coherence = compute_coherence_values(model=lsa_topics_for_coherence, corpus=corpus, dictionary=id2word, texts=df['plot'])
    print(f"Когеренция LSA (c_v): {lsa_coherence}")

    # # Визуализация LDA
    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.sklearn.prepare(lda_model_sklearn, vect_text, vect)
    # vis
    # # visualisation = pyLDAvis.gensim.prepare(lda_model_sklearn, corpus, id2word)
    # # pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')
