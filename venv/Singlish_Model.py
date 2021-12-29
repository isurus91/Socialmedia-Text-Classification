# ======= < Basic imports > ==============
import nltk.classify.util
import pandas as pd
import numpy  as np
import re
import random
import nltk
import tensorflow as tf
from keras.preprocessing.text import Tokenizer

from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder
from keras.models import load_model
from string import digits


def treatspecialdotuse(text):
    # tread special use of "." opprater
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|me|edu|lk)"
    digit = "([0-9])"
    sinhal = "([\u0D80-\u0DFFa])"
    PATTERN = r'[?|$|&|*|%|@|(|)|~|.|\'|\"|#|=|-|+]'
    unwanted = ["[", "?", "|", "$", "&", "*", "%", "@", "(", ")", "~", "#", "=", "-", "+", "]", "^"]
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    if "Ph.d" in text: text = text.replace("Ph.d", "Ph<prd>d<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(digit + "[.]" + digit, "\\1<prd>\\2", text)
    text = re.sub(sinhal + "[.]" + sinhal, "\\1<prd>\\2", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    text = text.replace("'", "")
    text = text.replace(";", "")
    text = text.replace(",", "")
    text = text.replace("`", "")
    sentences = text.split("<stop>")

    if (sentences[-1] == " "):
        sentences = sentences[:-1]
    clear_sentence = ''
    for single_sentence in sentences:
        # remove white spaceings
        single_sentence = single_sentence.strip()
        if (single_sentence.endswith('!')):
            clear_sentence += single_sentence.replace(single_sentence[len(single_sentence) - 1], ' ')
        elif (single_sentence.endswith('?')):
            clear_sentence += single_sentence.replace(single_sentence[len(single_sentence) - 1], ' ')
        elif (single_sentence.endswith('.')):
            clear_sentence += single_sentence.replace(single_sentence[len(single_sentence) - 1], ' ')
        else:
            clear_sentence += single_sentence
    wordbox = ''
    for word in clear_sentence.split():
        # remove URLS
        if (word[0] == 'w' and word[1] == 'w' and word[2] == 'w'):
            wordbox += ''
        elif (word[0] == 'h' and word[1] == 't' and word[2] == 't' and word[3] == 'p'):
            wordbox += ''
        # remove #hages
        elif (word[0] == '#'):
            wordbox += ''
        # remove anotations
        elif (word[0] == '@'):
            wordbox += ''
        else:
            letter_box = ''
            for letter in range(0, len(word)):
                if (word[letter].isalpha()):
                    # make lower case for english chars
                    letter_box += word[letter].lower()
                else:
                    letter_box += word[letter]
            wordbox += letter_box
        # add spacing between words
        wordbox += ' '

    # wordbox = re.sub(PATTERN, r'', wordbox)  # remove special char's
    final_format = str(wordbox).translate({ord(k): None for k in digits})
    for i in unwanted:
        final_format = final_format.replace(i, '')
    return final_format.replace("/", "")  # remove numbers
    # return re.sub(' +', ' ', wordbox).strip(str(wordbox).translate({ord(k): None for k in digits}))

#get_Sinhala_List
def get_Singlish_List():
    print("1 Singlish stop words")
    singlish_stop_words_list = ['dha', 'ya', 'mea', 'ea', 'haa', 'ma', 'bawa', 'nam', 'dhe', 'shdhaha', 'yi',
                                'shaha', 'lesa', 'hoo', 'aetha', 'nisa', 'meama', 'wa', 'gaena', 'wisin', 'wita',
                                'ema', 'thula', 'kota', 'yuthu', 'washayen', 'no', 'pilibadha', 'anuwa', 'eheth',
                                'pasu', 'eak', 'mean', 'sita', 'neatha', 'weani', 'nea', 'wada', 'bohoa', 'etha',
                                'naehea', 'thawath', 'bawata', 'pamanak', 'yam', 'samaga', 'matha', 'sae', 'namuth',
                                'peara', 'maha', 'weatha', 'pamana', 'samaga', 'nowea', 'magin', 'enam', 'weanuwen',
                                'yatathea', 'saema', 'pasuwa', 'thamayi', 'dhakwa', 'nowa', 'siyalu', 'yaeyi',
                                'aethaem', 'yanu', 'waenuwen', 'haema', 'pinisa', 'patan', 'sahitha', 'hari',
                                'aethulu', 'wagae', 'paridhi', 'pawa', 'thawa', 'samaharu', 'samahara', 'eath',
                                'wena', 'baewin', 'pahala', 'wadaath', 'haera', 'pita', 'eweni', 'aa', 'ehala',
                                'thulin', 'ekka', 'aadhi', 'mulu', 'sambandhayean', 'peawathi', 'oya', 'wath',
                                'pamani', 'wen', 'pura', 'ona', 'epa', 'ewita', 'hewath', 'yanuwen', 'meawaeni',
                                'ow', 'ethi', 'tika', 'ihatha', 'kara', 'kisiyam', 'aeyi', 'kisima', 'ara', 'heyin',
                                'kisidhu', 'hata', 'anek', 'idhiri', 'thek', 'waenuwata', 'haetiyata', 'yata',
                                'dhaeyi', 'mokadha', 'maghin', 'baew', 'an', 'ona', 'laga', 'uth', 'haraha',
                                'asala', 'baehaera', 'hoth', 'ekthara', 'uda', 'athurin', 'klhi', 'ayuru', 'noyek',
                                'atharin', 'ammea', 'aethuluwa', 'hariyatama', 'ethana', 'ehaata', 'kawara',
                                'wanaahi', 'osseama', 'naethinam', 'udhesa', 'naedha', 'baehae', 'misa', 'haeti',
                                'wata', 'bae', 'etharam', 'thawadhuratath', 'thawa', 'thuru', 'idhiriyae',
                                'edhiriyae', 'hariyata', 'ebadhu', 'naethdha', 'naedhdha', 'anea', 'mona',
                                'ethamath', 'ithamath', 'ethakota', 'aneakuth', 'awata', 'aetha', 'meabadhu',
                                'lath', 'thulata', 'kohomadha', 'kumana', 'lesin', 'badhu', 'oonaema', 'sahagatha',
                                'lagata', 'wethata', 'waenidhaa', 'onea', 'passea', 'thissea', 'saeti', 'baegin',
                                'sesu',
                                'bohoma', 'kotharam', 'aethath', 'yanuwen', 'koyi', 'upa', 'yae', 'kotharam', 'o',
                                'anith', 'onna', 'naethahoth', 'witharak', 'kawdha', 'hugak', 'naetheyi', 'pitatha',
                                'saema', 'nisama', 'passea', 'seaka', 'ewan', 'monawadha', 'samagama', 'thawadha',
                                'sadha', 'haebaeyi', 'atharin', 'lesata', 'dhigata', 'mandha', 'namuth', 'namudhu',
                                'eheanam', 'kepa', 'emagin', 'idhin', 'mokakdha', 'dhigea', 'witaka', 'meani',
                                'sambandhawa', 'nohoth', 'anna', 'monawadha', 'harima', 'naethinam', 'atharaata',
                                'pitathata', 'wan', 'yatin', 'wagea', 'nisath', 'pahalata', 'anya', 'eha', 'witama',
                                'vitama', 'noyekuth', 'withara',
                                'ihalin', 'iwatha', 'nomethiwa', 'hudeak', 'meaweni', 'waethin', 'weni', 'waewa',
                                'andhamin', 'meanma', 'was', 'kiipa', 'kochchra', 'methaenin', 'methena',
                                'idhiripita', 'purama', 'witaka', 'athishaya', 'dhepasa', 'peareadha', 'aethulata',
                                'atharea', 'yamkisi', 'aassithawa', 'thak', 'koa', 'uda', 'meatha', 'ehalata',
                                'dhenatamath', 'wenama', 'arabeaya', 'rahitha', 'pitin', 'paahea', 'aethinam',
                                'aeththam', 'atharam', 'kuwuruth', 'paridhdhen', 'watea', 'keabadhu', 'kothek',
                                'naththam', 'hindha', 'pasupasa', 'meha', 'ladheyi', 'paasa', 'pawa', 'perata',
                                'waediputa', 'aethin', 'ethek', 'vina', 'maedhin', 'idhiriyean', 'pitupasa',
                                'samaga', 'mae', 'la', 'maenawi', 'sampanna', 'athurean', 'meathram',
                                'obbata', 'nomilea', 'arabaya', 'abiyasa', 'idhiripasa', 'enamuth',
                                'mathin''issaraha', 'echchrara', 'ekko', 'behi', 'magin', 'magini', 'ketuwa',
                                'misak', 'eheamyi', 'hindha', 'idhin', 'lagin', 'asalin', 'dho''kelinma',
                                'virahitha', 'mathata', 'asalata', 'atharahura',
                                'mealagata', 'kethek', 'kola', 'ayyo', 'nomilea', 'ethanata', 'mulata', 'lu',
                                'samagi', 'che', 'nithi', 'wenidha', 'apoyi', 'ochchrata', 'aaubowan', 'huga',
                                'wanathuru', 'pamanata', 'pahathata', 'ewaka', 'pasakin', 'aethulea', 'yatathata',
                                'yabadha', 'nan',
                                'shik', 'shek', 'usi', 'anan', 'manan', 'ananmanan', 'ahaa', 'adhdharata',
                                'monayam', 'athana', 'aethule', 'aethulen', 'pahathin', 'virahithawa', 'aayi',
                                'enayin', 'samagin', 'obbehi', 'othana', 'ammo', 'sha', 'neathahoth', 'atharamaga',
                                'asabada', 'yabadhawa', 'seakwa', 'atharamaga', 'obben', 'adhdhara',
                                'achchrara', 'dhoho', 'chi', 'chik', 'adho', 'soshae', 'ethira', 'koyibata', 'ohea',
                                'oh', 'adhomeyi', 'uhu', 'uu', 'u', 'otharam', 'abiyasin', 'appachchiyea', 'ado',
                                'aane', 'ane', 'hindhama', 'ha', 'hapura', 'latha', 'ohe', 'an', 'ankisi', 'hurea',
                                'uuyi', 'adhdharin', 'embala', 'abiyasa', 'abiyasata',
                                'uuh', 'shi', 'sisi', 'koyibata', 'satapata', 'abiyesa', 'heleyiya', 'sata',
                                'satata'

                                ]
    return singlish_stop_words_list


def remove_html(text):
    text = text.replace('<br />', ' ')
    if (text.find('<') != -1 and text.rfind('>') != -1):
        if (len(text) == text.rfind('>')):  # check string end with html tag
            text = text[0: text.find('<'):]
        elif (text.find('<') == 0):  # check string start with html tag
            text = text[text.rfind('>') + 1::]
        else:
            text = text[0: text.find('<'):] + text[text.rfind('>') + 1::]
    return text


def remove_stopwords(tokens):
    filtered_tokens = [token for token in tokens if token not in get_Singlish_List()]
    return filtered_tokens


def get_train_test():
    print("get_train_test")
    # 1) Read excel file and store the data to list
    all_sinhala_strings = ""
    all_sinhala_string_list = []
    all_sinhala_string_count_list = []
    all_sinhala_tag_list = []

    sinhala_N_strings = ""
    sinhala_N_string_list = []
    sinhala_N_string_count_list = []

    sinhala_O_strings = ""
    sinhala_O_string_list = []
    sinhala_O_string_count_list = []

    sinhala_H_strings = ""
    sinhala_H_string_list = []
    sinhala_H_string_count_list = []

    data = pd.read_csv("Singlish_Data.csv")
    len(data)

    comment_list = list()
    comment_list_sort = list()
    for index, row in data.iterrows():
        comment_list.append(len(row['Coment'].split()))
        comment_list_sort.append(len(row['Coment'].split()))
    comment_list_sort.sort()

    for index, row in data.iterrows():
        word = row['Coment']
        separator = " "
        clean_word = (separator.join(remove_stopwords(row['Coment'].split())))
        all_sinhala_string_list.append(clean_word)
        all_sinhala_tag_list.append(row['Status'])
        if (row['Status'] == "Neutral "):
            sinhala_N_string_list.append(clean_word)
            sinhala_N_string_count_list.append(len(clean_word.split()))
        if (row['Status'] == "offensive " or row['Status'] == "Offensive "):
            sinhala_O_string_list.append(clean_word)
            sinhala_O_string_count_list.append(len(clean_word.split()))
        if (row['Status'] == "Hate"):
            sinhala_H_string_list.append(clean_word)
            sinhala_H_string_count_list.append(len(clean_word.split()))

    all_sinhala_strings = " ".join(all_sinhala_string_list)
    sinhala_N_strings = " ".join(sinhala_N_string_list)
    sinhala_O_strings = " ".join(sinhala_O_string_list)
    sinhala_O_strings = " ".join(sinhala_O_string_list)

    random.seed(50)
    random_hate_list = random.sample(sinhala_H_string_list, 500)
    random_hate_list = random.sample(random_hate_list + random_hate_list, 800)
    random_offensive_list = random.sample(sinhala_O_string_list, 600)
    random_offensive_list = random.sample(random_offensive_list + random_offensive_list, 800)
    random_netural_list = random.sample(sinhala_N_string_list, 800)

    random_all_list = random_hate_list + random_offensive_list + random_netural_list

    all_sinhala_strings = " ".join(random_all_list)

    sinhala_tokens = nltk.word_tokenize(all_sinhala_strings)
    print("number words in the courpes : " + str(len(sinhala_tokens)))
    print("number unic words in the courpes : " + str(len(set(sinhala_tokens))))

    Hate_header_list = []
    Netural_header_list = []
    Offincive_header_list = []

    for i in range(0, 800):
        Hate_header_list.append(3)
        Netural_header_list.append(1)
        Offincive_header_list.append(2)

    random_all_header_list = Hate_header_list + Offincive_header_list + Netural_header_list

    x = random_hate_list
    y = Hate_header_list
    X_H_train, X_H_test, y_H_train, y_H_test = train_test_split(x, y, test_size=0.2, random_state=50)

    x = random_offensive_list
    y = Offincive_header_list
    X_O_train, X_O_test, y_O_train, y_O_test = train_test_split(x, y, test_size=0.2, random_state=50)

    x = random_netural_list
    y = Netural_header_list
    X_N_train, X_N_test, y_N_train, y_N_test = train_test_split(x, y, test_size=0.2, random_state=50)

    X_train = X_H_train + X_O_train + X_N_train
    X_test = X_H_test + X_O_test + X_N_test
    y_train = y_H_train + y_O_train + y_N_train
    y_test = y_H_test + y_O_test + y_N_test

    print("end excel reading")
    return X_train, X_test, y_train, y_test


def set_tf_idf_fordataset():
    print("tf idf")
    X_train, X_test, y_train, y_test = get_train_test()
    num_words = 30001
    tokenizer = Tokenizer(num_words=num_words, oov_token="unk", lower=False, split=' ')
    # tokenizer.fit_on_texts((X_train+X_test))
    tokenizer.fit_on_texts((X_train))

    x_train = np.array(tokenizer.texts_to_sequences(X_train))
    x_valid = np.array(tokenizer.texts_to_sequences(X_test))
    # add padding in the end of the sentence when len is 50
    x_train = pad_sequences(x_train, padding='post', maxlen=50)
    x_valid = pad_sequences(x_valid, padding='post', maxlen=50)

    # lable encorder
    le = LabelEncoder()

    train_labels = le.fit_transform(y_train)
    train_labels = np.asarray(tf.keras.utils.to_categorical(train_labels))
    # print(train_labels)
    valid_labels = le.transform(y_test)
    valid_labels = np.asarray(tf.keras.utils.to_categorical(valid_labels))

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, train_labels))
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, valid_labels))

    count = 0
    print('======Train dataset ====')
    for value, label in train_ds:
        count += 1
        print(value, label)
        if count == 3:
            break
    count = 0
    print('======Validation dataset ====')
    for value, label in valid_ds:
        count += 1
        print(value, label)
        if count == 3:
            break

    return train_ds, valid_ds, x_valid, valid_labels, tokenizer


def singlish_predection_model():
    train_ds, valid_ds, x_valid, valid_labels, tokenizer = set_tf_idf_fordataset()
    max_features = 30001
    embedding_dim = 64
    sequence_length = 50
    epochs = 100

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(max_features, embedding_dim, input_length=sequence_length, \
                                        embeddings_regularizer=regularizers.l2(0.0005)))

    model.add(tf.keras.layers.Conv1D(128, 3, activation='relu', \
                                     kernel_regularizer=regularizers.l2(0.0005), \
                                     bias_regularizer=regularizers.l2(0.0005)))
    model.add(tf.keras.layers.Dense(128, activation='relu', \
                                    kernel_regularizer=regularizers.l2(0.0005), \
                                    bias_regularizer=regularizers.l2(0.0005)))
    model.add(tf.keras.layers.Dense(128, activation='relu', \
                                    kernel_regularizer=regularizers.l2(0.0005), \
                                    bias_regularizer=regularizers.l2(0.0005)))
    model.add(tf.keras.layers.GlobalMaxPooling1D())

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(3, activation='sigmoid', \
                                    kernel_regularizer=regularizers.l2(0.001), \
                                    bias_regularizer=regularizers.l2(0.001), ))

    model.summary()
    # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='Nadam', metrics=["CategoricalAccuracy"])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Nadam(learning_rate=.0001), metrics=["CategoricalAccuracy"])

    history = model.fit(train_ds.shuffle(2000).batch(128),
                        epochs=epochs,
                        validation_data=valid_ds.batch(128),
                        verbose=1)

    model.save('models/Singlish_Clasification_model.h5')

    y_pred = model.predict(x_valid)

    cnf_metrics = metrics.confusion_matrix(valid_labels.argmax(axis=1), y_pred.argmax(axis=1))
    print(classification_report(valid_labels.argmax(axis=1), y_pred.argmax(axis=1)))
    print("Accuracy : ", metrics.accuracy_score(valid_labels.argmax(axis=1), y_pred.argmax(axis=1)))

def get_singlish_predection_model(text):
    separator = " "
    clean_word = (separator.join(remove_stopwords(text.split())))
    clean_text = treatspecialdotuse(remove_html(clean_word))

    model = load_model('models/Singlish_Clasification_model.h5')
    a, b, c, d, tokenizer = set_tf_idf_fordataset()
    x_test = np.array(
        tokenizer.texts_to_sequences(
            # ('hena gahapan pake lawke huththo', 'kalaguna salakeema uthum kriyawak', 'kalakanni media karayo shit')))
            ([clean_text])))
    x_test = pad_sequences(x_test, padding='post', maxlen=50)

    print("Generate predictions for all samples")
    predictions = model.predict(x_test)
    predict_results = predictions.argmax(axis=1)
    print(predictions.tolist)
    print(predict_results)
    return predict_results,predictions