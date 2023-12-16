import streamlit as st
from sre_parse import Tokenizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Декоратор @st.cache говорит Streamlit, что модель нужно загрузить только один раз, чтобы избежать утечек памяти
@st.cache_data
def load_model(model_checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()
    return model

# Загружаем предварительно обученную модель
model = load_model(model_checkpoint)

# Функция для работы с моделью
def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

# Собственно наш веб-интрфейс
st.title('Определение токсичности приведенной фразы')

text = st.text_input("Пожалуйста, напишите свою фразу:")

# При нажатии кнопки Enter
if st.button('Enter'):
    # ваш код для обработки ввода текста
    result = text2toxicity(text, True)
    st.write("Вероятность того, что текст содержит токсичные выражения составляет: ", result)







