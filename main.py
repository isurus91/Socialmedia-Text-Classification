# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from flask import Flask
from flask_restful import Resource, Api
from Sinhala_Model import sinhal_predection_model,get_sinhal_predection_model
from Sinhala_Model import  get_sinhal_predection_model
from Singlish_Model import  get_singlish_predection_model
from Singlish_Model import singlish_predection_model
from emoji_model import is_Singlish


app = Flask(__name__)
api = Api(app)


@app.route('/predection/<text>')
def get_predection_result(text):
    type = ""
    # get emojis
    language_type, hate_emojis, offincive_emojis, normal_emojis = is_Singlish(text)
    if language_type==False:
        print("Activated sinhala model")
        #Status, Probility = get_test_labels(text)
        Status, Probility = get_sinhal_predection_model(text)
        print(Status)
        print(Probility)
        if (Status.tolist()[0] == 1):
            type = "Neutral"
        if (Status.tolist()[0] == 2):
            type = "Offensive"
        if (Status.tolist()[0] == 3):
            type = "Hate"

        #x=[]
        #x.append(Probility.values.tolist())
        #aaaa= Convert(str(x)[2:-2])
        #print(aaaa)
        return {'Data': text, 'StatusType': type, 'StatusID': Status.tolist(), 'Probility': Probility.tolist(),
                'IsSinglish': language_type, 'hate_emojis_list': hate_emojis, 'offincive_emojis_list': offincive_emojis,
                'normal_emojis_list': normal_emojis}
    else:
        print("Activated singish model")
        #get model predection
        Status, Probility = get_singlish_predection_model(text)
        print(Status)
        print(Probility)
        if(Status.tolist()[0]==0):
            type="Neutral"
        if(Status.tolist()[0]==1):
            type="Offensive"
        if(Status.tolist()[0]==2):
            type="Hate"
        return {'Data': text,'StatusType':type,'StatusID':Status.tolist(),'Probility':Probility.tolist(),'IsSinglish':language_type,'hate_emojis_list':hate_emojis,'offincive_emojis_list':offincive_emojis,'normal_emojis_list':normal_emojis}





@app.route('/ClassificationModel/<text>')
def get_sinhala_status(text):
    print(text)
    predictions, predict_results = get_singlish_predection_model(text)#get_sinhal_predection_model(text)
    print("********************")
    print(predictions.tolist())
    print("********************")
    print(predict_results.tolist())
    if (predictions.tolist()[0] == 1):
        type = "Neutral"
    if (predictions.tolist()[0] == 2):
        type = "Offensive"
    if (predictions.tolist()[0] == 3):
        type = "Hate"
    print("End")
    return ("Hi" + text);

def Convert(string):
  return [int(i) for i in list(string.split(" "))]

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    #sinhal_predection_model()
    #singlish_predection_model()
    a,b,c,d,=is_Singlish("ade mata hinai hutthooo 😀 😃 😄 😁 😄 😂 🤣 😇 😆 😁 😄 😃 😀 😜 😛 😋 🤩 😍 🥰 🤪 😝 😘 😆 😁 😄 😃 😀 😅 💩 🖕 🤦 ☸ 👍")
    print(a)
    print(b)
    print(c)
    print(d)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
