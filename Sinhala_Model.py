import nltk.classify.util
import numpy as np
import pandas as pd
import re
import random
import nltk
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
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

#========= Classifiers ==========================================
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,ExtraTreesClassifier,BaggingClassifier
from sklearn.svm import SVC, LinearSVC
#========== <Count Vectorization,feature extraction and summaries > ======================
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import  train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error
# =========== < End > ===================
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

def get_Sinhala_List():
    sinhala_stop_words_list = ['බොහොම', 'කොතරම්', 'ඇතත්', 'යනුවෙනි', 'කොයි', 'උප', 'යැ', 'කෙතරම්', 'ඕ', 'අනිත්',
                                   'ඔන්න',
                                   'නැතහොත්', 'විතරක්', 'කවුද', 'හුඟක්', 'නැතැයි', 'පිටත', 'සැම', 'නිසාම', 'පස්සේ',
                                   'සේක',
                                   'එවන්'
            , 'මොනවද', 'සමගම', 'තවද', 'සඳ', 'හැබැයි', 'අතුරින්', 'ලෙසට', 'දිගට', 'මන්ද', 'නමුදු', 'එහෙනම්', 'කිහිප',
                                   'එමගින්', 'ඉදින්', 'මොකක්ද', 'දිගේ', 'විටෙක', 'මෙනි', 'සම්බන්ධව', 'නොහොත්', 'අන්න',
                                   'මොනවා', 'හරිම', 'නැතිනම්', 'අතරට'
            , 'පිටතට', 'වන්', 'යටින්', 'වාගේ', 'නිසාත්', 'පහළට', 'අන්‍ය', 'එහා', 'විටම', 'නොයෙකුත්', 'විතර', 'ද', 'ය',
                                   'මේ',
                                   'ඒ', 'හා', 'ම', 'බව', 'නම්', 'දී', 'සඳහා', 'යි', 'සහ', 'ලෙස', 'හෝ', 'ඇත', 'නිසා',
                                   'මෙම',
                                   'ව', 'ගැන', 'විසින්', 'විට', 'එම', 'තුළ', 'කොට'
            , 'යුතු', 'වශයෙන්', 'නො', 'පිළිබඳ', 'අනුව', 'එහෙත්', 'පසු', 'එක්', 'මෙන්', 'සිට', 'නැත', 'වැනි', 'නෑ',
                                   'වඩා',
                                   'බොහෝ'
            , 'ඉතා', 'නැහැ', 'තවත්', 'බවට', 'පමණක්', 'යම්', 'සමඟ', 'මත', 'සේ', 'නමුත්', 'පෙර', 'මහා', 'වෙත', 'පමණ',
                                   'සමග',
                                   'නොවේ', 'මගින්', 'එනම්', 'වෙනුවෙන්', 'යටතේ', 'සෑම', 'පසුව', 'තමයි', 'දක්වා', 'නොව'
            , 'යැයි', 'ඇතැම්', 'යනු', 'වෙනත්', 'හැම', 'පිණිස', 'පටන්', 'සහිත', 'හරි', 'ඇතුළු', 'වගේ', 'පරිදි', 'පවා',
                                   'තව',
                                   'සමහර', 'ඒත්', 'වෙන', 'බැවින්', 'පහළ', 'වඩාත්', 'හැර', 'පිට', 'එවැනි', 'කිසි', 'ආ',
                                   'ඉහළ', 'අනෙක්'
            , 'තුළින්', 'එක්ක', 'ආදී', 'මුළු', 'සම්බන්ධයෙන්', 'පැවති', 'ඔය', 'වත්', 'පමණි', 'වෙන්', 'පුරා', 'ඕනෑ',
                                   'එපා',
                                   'එවිට', 'හෙවත්', 'යනුවෙන්'
            , 'මෙවැනි', 'ඔව්', 'ඉතින්', 'ටික', 'ඉහත', 'කරා', 'කිසියම්', 'ඇයි', 'කිසිම', 'අර', 'හෙයින්', 'කිසිදු', 'හට',
                                   'අනික්', 'ඉදිරියට', 'තෙක්', 'වෙනුවට', 'හැටියට', 'යට', 'දැයි', 'මොකද'
            , 'මඟින්', 'බැව්', 'අන්', 'ඕන', 'ළඟ', 'යුත්', 'හරහා', 'අසල', 'බැහැර', 'හොත්', 'එක්තරා', 'උඩ', 'අයුරින්',
                                   'කල්හි', 'අයුරු', 'නොයෙක්', 'අතරින්', 'අම්මේ', 'ඇතුළුව', 'හරියටම', 'මෙච්චර', 'එතන',
                                   'එහාට', 'කවර', 'වනාහි', 'ඔස්සේ', 'නැත්නම්'
            , 'උදෙසා', 'යුක්ත', 'නේද', 'බැහැ', 'මිස', 'හැටි', 'වටා', 'බෑ', 'එතරම්', 'තවදුරටත්', 'තුරු', 'ඉදිරියේ',
                                   'හරියට',
                                   'එබඳු', 'නැද්ද', 'අනේ', 'මොන', 'ඉතාමත්', 'එතකොට', 'අනෙකුත්', 'අවට', 'ඈත', 'මෙබඳු',
                                   'ලත්',
                                   'තුළට', 'කොහොමද', 'කුමන'
            , 'ලෙසින්', 'බඳු', 'ඕනෑම', 'සහගත', 'ළඟට', 'වෙතට', 'වැනිදා', 'ඕනෙ', 'තිස්සේ', 'සැටි'
            , 'බැගින්', 'සෙසු', 'ඉහළින්', 'ඉවත', 'නොමැතිව', 'හුදෙක්', 'මෙවන්', 'වෙතින්', 'වෙනි', 'වේවා', 'අන්දමින්',
                                   'මෙන්න', 'වස්', 'කීප', 'කොච්චර', 'මෙතන', 'ඉදිරිපිට', 'පුරාම', 'විටක', 'අතිශය'
            , 'දෙපස', 'පෙරේදා', 'ඇතුළට', 'අතරේ', 'යම්කිසි', 'ආශ්‍රිතව', 'තාක්', 'කෝ', 'උඩට', 'මෑත', 'ඉහළට', 'දැනටමත්',
                                   'වෙනම', 'අරබයා', 'රහිත', 'පිටින්', 'පාහේ', 'ඇත්නම්', 'අතරම', 'කවුරු'
            , 'පරිද්දෙන්', 'වටේ', 'කෙබඳු', 'කොතෙක්', 'නැතත්', 'හින්දා', 'පසුපස', 'මෙහා', 'ලදැයි', 'පාසා', 'පෙරට',
                                   'වැඩිපුර',
                                   'ඈතින්', 'එතෙක්', 'විනා'
            , 'මැදින්', 'ඉදිරියෙන්', 'පිටුපස', 'සමඟම', 'මැ', 'ල', 'මැනවි', 'සම්පන්න', 'අතුරෙන්', 'මෙතරම්', 'ඔබ්බට',
                                   'නොමිලේ', 'අරභයා', 'අබියස', 'ඉදිරිපස', 'එනමුත්', 'මතින්', 'ඉස්සරහ', 'එච්චර'
            , 'එක්කෝ', 'හෙබි', 'මගිනි', 'කැටුව', 'මිසක්', 'එහෙමයි', 'හින්ද', 'ඉඳන්', 'ළඟින්', 'අසලින්', 'දෝ', 'කෙළින්ම',
                                   'විරහිත', 'මතට', 'අසලට', 'අතරතුර', 'මීළඟ', 'කෙතෙක්', 'කොල', 'අයියෝ', 'නොමිලයේ',
                                   'එතනට',
                                   'මුලට', 'ලු',
                                   'සමගින්', 'චී', 'නිති', 'වෙනිදා', 'අපොයි', 'ඔච්චර', 'ආයුබෝවන්', 'හුඟ', 'වනතුරු'
            , 'පමණට', 'පහතට', 'එවක', 'පසෙකින්', 'ඇතුළේ', 'යටතට', 'යාබද', 'නන්', 'ෂික්', 'උසි', 'මනන්', 'අහා', 'අද්දරට',
                                   'මොනයම්', 'අතන', 'ඇතුළෙන්', 'පහතින්', 'විරහිතව', 'ආයි', 'එනයින්', 'සමඟින්',
                                   'ඔබ්බෙහි',
                                   'ඔතන', 'අම්මෝ', 'ෂා'
            , 'නැතොත්', 'අතරමඟ', 'අසබඩ', 'යාබදව', 'සේක්වා', 'අතරමග', 'ඔබ්බෙන්', 'අද්දර', 'අච්චර', 'දෝහෝ', 'චිහ්', 'අඳෝ',
                                   'සොඳේ', 'ඉතිර', 'කොයිබට', 'ඕහ්', 'අඳෝමැයි', 'උහ්', 'අඩෝ', 'ඔතරම්', 'අබියසින්',
                                   'අප්පච්චියේ', 'ආනේ', 'හින්දම'
            , 'හාපුරා', 'ලත', 'ඔහෝ', 'අන්කිසි', 'හුරේ', 'ඌයි', 'අද්දරින්', 'එම්බල', 'අබියසට', 'ඌහ්', 'ෂිහ්', 'සීසී',
                                   'කොයිබ', 'සටපට', 'අබියෙස', 'හෙලෙයියා', 'සටසට']

    return sinhala_stop_words_list

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
    filtered_tokens = [token for token in tokens if token not in get_Sinhala_List()]
    return filtered_tokens

def get_train_test():
    print("get_train_test")
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

    # read CSV
    data = pd.read_csv("Sinhala_Data.csv")

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
        if (row['Status'] == "Neutral"):
            sinhala_N_string_list.append(clean_word)
            sinhala_N_string_count_list.append(len(clean_word.split()))
        if (row['Status'] == "offensive"):
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
    random_hate_list = random.sample(sinhala_H_string_list, 400)
    random_hate_list = random_hate_list + random_hate_list
    random_offensive_list = random.sample(sinhala_O_string_list, 400)
    random_offensive_list = random_offensive_list + random_offensive_list
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
        #print(value, label)
        if count == 3:
            break
    count = 0
    print('======Validation dataset ====')
    for value, label in valid_ds:
        count += 1
        #print(value, label)
        if count == 3:
            break

    return train_ds, valid_ds, x_valid, valid_labels, tokenizer

def sinhal_predection_model():
    print("hate model")
    train_ds, valid_ds, x_valid, valid_labels, tokenizer = set_tf_idf_fordataset()
    max_features = 30001
    embedding_dim = 64
    sequence_length = 50

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(max_features, embedding_dim, input_length=sequence_length, \
                                        embeddings_regularizer=regularizers.l2(0.0005),
                                        embeddings_initializer="variance_scaling"))

    model.add(tf.keras.layers.Conv1D(128, 3, activation='linear', \
                                     kernel_regularizer=regularizers.l2(0.0005), \
                                     bias_regularizer=regularizers.l2(0.0005)))
    model.add(tf.keras.layers.Dense(128, activation='linear', \
                                    kernel_regularizer=regularizers.l2(0.0005), \
                                    bias_regularizer=regularizers.l2(0.0005)))
    model.add(tf.keras.layers.Dense(128, activation='linear', \
                                    kernel_regularizer=regularizers.l2(0.0005), \
                                    bias_regularizer=regularizers.l2(0.0005)))

    # https://deeplizard.com/learn/video/ZjM_XQa5s6s
    model.add(tf.keras.layers.GlobalMaxPooling1D())

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(3, activation='exponential', \
                                    kernel_regularizer=regularizers.l2(0.001), \
                                    bias_regularizer=regularizers.l2(0.001), ))

    model.summary()
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Nadam(learning_rate=.0001), metrics=["CategoricalAccuracy"])

    history = model.fit(train_ds.shuffle(2000).batch(128),
                        epochs=60,
                        validation_data=valid_ds.batch(128),
                        verbose=1)

    model.save('models/Sinhala_Clasification_model.h5')

    y_pred = model.predict(x_valid)
    # cnf_metrics = metrics.confusion_matrix(valid_labels,y_pred)

    cnf_metrics = metrics.confusion_matrix(valid_labels.argmax(axis=1), y_pred.argmax(axis=1))
    print(classification_report(valid_labels.argmax(axis=1), y_pred.argmax(axis=1)))
    print("Accuracy : ", metrics.accuracy_score(valid_labels.argmax(axis=1), y_pred.argmax(axis=1)))


def get_sinhal_predection_model(text):
    print("Incomming text is : "+text)
    separator = " "
    clean_word = (separator.join(remove_stopwords(text.split())))
    clean_text = treatspecialdotuse(remove_html(clean_word))

    model = load_model('models/Sinhala_Clasification_model.h5')
    a, b, c, d, tokenizer = set_tf_idf_fordataset()
    x_test = np.array(
        tokenizer.texts_to_sequences(
            # ('hena gahapan pake lawke huththo', 'kalaguna salakeema uthum kriyawak', 'kalakanni media karayo shit')))
            ([clean_text])))
    x_test = pad_sequences(x_test, padding='post', maxlen=50)

    print("Generate predictions for all samples")
    predictions = model.predict(x_test)
    predict_results = predictions.argmax(axis=1)
    return predict_results,predictions
