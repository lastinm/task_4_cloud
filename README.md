# task_4_cloud

WEB приложение машинного обучения развернуто в облаке Streamlit (https://streamlit.io).

# Об использованной модели машинного обучения

В качестве модели распознавания использована следующая модель:
https://huggingface.co/cointegrated/rubert-tiny-toxicity

Это коинтегрированная модель Руберта-Тайни, 
доработанная для классификации токсичности и неуместности коротких неформальных русских текстов, 
таких как комментарии в социальных сетях.

Достоинство данной модели - это ее очень малый размер, всего 47,2 Мб.
model.safetensors: 100%|██████████| 47.2M

При таком невероятно малом размере, модель сохраняет возможность находить токсичность в переданных текстах.

# Работа приложения

На главной странице сайта: https://task4cloud.streamlit.app/ расположено поле для ввода текста:
"Пожалуйста, напишите свою фразу:"

Для распознавания укажем такой текст (прощу прощения за грубость, но нужно проверять работу):
"Эта лодка дырявая калоша".

Отправляем текст нажатием кнопки "Enter".
Получаем ответ: "Вероятность того, что текст содержит токсичные выражения составляет: 0.978443726230604".
Как видим, модель отработала верно. С вероятностью 0.998 было определено, что текст токсичен.

Теперь введем: "Ты мой ласковый и нежный зверь".
Получаем ответ: Вероятность того, что текст содержит токсичные выражения составляет: 0.024751935718452467".
Модель снова отработала успешно, определив вероятность 0,024 того, что текст является токсичным.

