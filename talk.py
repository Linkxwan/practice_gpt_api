import g4f
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import texts


print("Loading...")
question = "какие специальности есть в колледже?"

vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(texts)
print("Data is done!")


def context_generation():
    return texts[cosine_similarity(text_vectors, vectorizer.transform([question])).argmax()]


def GPT_talk():
    try:
        # шаблон
        template = f"""Ты - полезный ИИ ассистент для нашего колледжа комтехно.
        Используйте следующие фрагменты контекста (Context), чтобы ответить на вопрос в конце (Question).
        Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь придумывать ответ.
        Сначала убедитесь, что прикрепленный текст имеет отношение к вопросу.
        Если текст не имеет отношения к вопросу, просто скажите, что текст не имеет отношения.
        Используйте максимум три предложения. Держите ответ как можно более кратким.
        Context: {context_generation():}
        Question: {question}
        Helpful Answer:"""

        # обращяемся по api к chat GPT
        response = g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": template}],
            stream=True,
        )

        print(question)

        # Выводим ответ нейросети
        for message in response:
            print(message, flush=True, end='')


    except Exception as e:
        # Обработка ошибок
        print(f"Произошла ошибка: {e}")
        

def main():
    GPT_talk()


if __name__ == "__main__":
    main()