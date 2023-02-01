import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64
from keras.preprocessing import image
from PIL import Image, ImageOps
import cv2
import os
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from skimage.io import imread, imshow
from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

import tensorflow_hub as hub
from tensorflow.keras.activations import softmax

st.markdown('<h1 style="color:black;">CNN Sick Mushroom Image classification model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> Oyster, Pleurotus, Portobello, Shiitake, Winter</h3>', unsafe_allow_html=True)

# background image to streamlit

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('../streamlit/mushroom.webp')


file_uploded= st.file_uploader('Insert image for classification', type=['png','jpg', 'jpeg'])
c1, c2= st.columns(2)
if file_uploded is not None:
    im= Image.open(file_uploded)
    img= np.asarray(im)
    img= cv2.resize(img,(150, 150))
    img= preprocess_input(img)
    img= np.expand_dims(img, axis=0)
    c1.header('Input Image')
    c1.image(im)
    c1.write(img.shape)
    
      #load weights of the trained model.
    classifier_model = tf.keras.models.load_model(r'../streamlit/pages/sickmushroomcnn.hdf5')
    shape = ((150, 150, 3))
    model = tf.keras.Sequential(hub.KerasLayer(classifier_model, input_shape=shape))
    # test_image=image.resize((150, 150))
    # test_image=preprocessing.image.img_to_array(test_image)
    # test_image = test_image/255.0
    # test_image = np.expand_dims(test_image, axis=0)
    class_names = ['oyster_blue',
                  'oyster_brown',
                 'pleurotus_blue',
                 'pleurotus_brown',
                 'pleurotus_white',
                 'portobello_blue', 
                 'portobello_brown', 
                 'portobello_white',
                 'shiitake_blue',
                 'shiitake_brown',
                 'shiitake_white', 
                 'winter_black', 
                 'winter_blue', 
                 'winter_white']
    predictions = model.predict(img)
    # scores = tf.nn.softmax(predictions[0])
    # scores = scores.numpy()
    pred_class = class_names[predictions.argmax()]
    c2.header('Output')
    c2.subheader(f'Predicted class : {pred_class}')
    
    if pred_class == 'shiitake_blue' :
        c2.write('표고버섯 푸른곰팡이병 \n - 재배사 바닥을 콘크리트로 개선하여 병해충의 서식지를 제거한다.\n - 버섯파리의 방제\n - 정확히 60-65도에서 8-12시간동안 살균한다.\n - 균상 위에 버섯잔유물이 부패될 때 2차 비기생성 병원균이 발생하므로 수확시 균상 정리가 필수다. \n - 폐상퇴비는 살균을 한 후 폐상을 하며 되도록 먼 곳으로 이동시켜 2차 오염을 방지한다. \n - 2회 살균시 1회 살균과 같이 살균시간을 과다하게 하는경우 병해가 더 심해질 수 있으니 주의한다. \n - 2회이상 살균시 볏집다발을 뒤집거나 상면의 톱밥을 제거한 후 살균하고 후발효기간동안 환기를 자주한다. \n - 균사생장 기간동안에 볏짚다발재배에서는 살균전 균상표면에 벤레이트, 판마쉬, 스포르곤수화제를 1000배액으로 평당 5~6g을 살포한다.')
    
    elif pred_class == 'oyster_blue' :
        c2.write('느타리 푸른곰팡이병 \n - 재배사 바닥을 콘크리트로 개선하여 병해충의 서식지를 제거한다.\n - 버섯파리의 방제\n - 정확히 60-65도에서 8-12시간동안 살균한다.\n - 균상 위에 버섯잔유물이 부패될 때 2차 비기생성 병원균이 발생하므로 수확시 균상 정리가 필수다. \n - 폐상퇴비는 살균을 한 후 폐상을 하며 되도록 먼 곳으로 이동시켜 2차 오염을 방지한다. \n - 2회 살균시 1회 살균과 같이 살균시간을 과다하게 하는경우 병해가 더 심해질 수 있으니 주의한다. \n - 2회이상 살균시 볏집다발을 뒤집거나 상면의 톱밥을 제거한 후 살균하고 후발효기간동안 환기를 자주한다. \n - 균사생장 기간동안에 볏짚다발재배에서는 살균전 균상표면에 벤레이트, 판마쉬, 스포르곤수화제를 1000배액으로 평당 5~6g을 살포한다.')
    
    elif pred_class == 'pleurotus_blue' :
        c2.write('큰느타리버섯 푸른곰팡이병 \n - 재배사 바닥을 콘크리트로 개선하여 병해충의 서식지를 제거한다.\n - 버섯파리의 방제\n - 정확히 60-65도에서 8-12시간동안 살균한다.\n - 균상 위에 버섯잔유물이 부패될 때 2차 비기생성 병원균이 발생하므로 수확시 균상 정리가 필수다. \n - 폐상퇴비는 살균을 한 후 폐상을 하며 되도록 먼 곳으로 이동시켜 2차 오염을 방지한다. \n - 2회 살균시 1회 살균과 같이 살균시간을 과다하게 하는경우 병해가 더 심해질 수 있으니 주의한다. \n - 2회이상 살균시 볏집다발을 뒤집거나 상면의 톱밥을 제거한 후 살균하고 후발효기간동안 환기를 자주한다. \n - 균사생장 기간동안에 볏짚다발재배에서는 살균전 균상표면에 벤레이트, 판마쉬, 스포르곤수화제를 1000배액으로 평당 5~6g을 살포한다.')
    
    elif pred_class == 'portobello_blue' :
        c2.write('양송이버섯 푸른곰팡이병 \n - 재배사 바닥을 콘크리트로 개선하여 병해충의 서식지를 제거한다.\n - 버섯파리의 방제\n - 정확히 60-65도에서 8-12시간동안 살균한다.\n - 균상 위에 버섯잔유물이 부패될 때 2차 비기생성 병원균이 발생하므로 수확시 균상 정리가 필수다. \n - 폐상퇴비는 살균을 한 후 폐상을 하며 되도록 먼 곳으로 이동시켜 2차 오염을 방지한다. \n - 2회 살균시 1회 살균과 같이 살균시간을 과다하게 하는경우 병해가 더 심해질 수 있으니 주의한다. \n - 2회이상 살균시 볏집다발을 뒤집거나 상면의 톱밥을 제거한 후 살균하고 후발효기간동안 환기를 자주한다. \n - 균사생장 기간동안에 볏짚다발재배에서는 살균전 균상표면에 벤레이트, 판마쉬, 스포르곤수화제를 1000배액으로 평당 5~6g을 살포한다.')
    
    elif pred_class == 'winter_blue':
        c2.write('팽이버섯 푸른곰팡이병 \n - 재배사 바닥을 콘크리트로 개선하여 병해충의 서식지를 제거한다.\n - 버섯파리의 방제\n - 정확히 60-65도에서 8-12시간동안 살균한다.\n - 균상 위에 버섯잔유물이 부패될 때 2차 비기생성 병원균이 발생하므로 수확시 균상 정리가 필수다. \n - 폐상퇴비는 살균을 한 후 폐상을 하며 되도록 먼 곳으로 이동시켜 2차 오염을 방지한다. \n - 2회 살균시 1회 살균과 같이 살균시간을 과다하게 하는경우 병해가 더 심해질 수 있으니 주의한다. \n - 2회이상 살균시 볏집다발을 뒤집거나 상면의 톱밥을 제거한 후 살균하고 후발효기간동안 환기를 자주한다. \n - 균사생장 기간동안에 볏짚다발재배에서는 살균전 균상표면에 벤레이트, 판마쉬, 스포르곤수화제를 1000배액으로 평당 5~6g을 살포한다.')
        
    elif pred_class == 'winter_black':
        c2.write('팽이버섯 세균성검은썩음병 \n - 병원균을 전파하는 버섯파리와 응애의 방제철저. \n - 관수용 지하수 저수조를 정기적으로 클로 로칼키 3000~5000배 액으로 세척 및 소독하며 폐상 소독을 철저히 한다. \n - 70℃의 건열 또는 습열로 10∼12시간 살균하거나 재배사 내에 포르말린을 ㎥당 30cc로 훈 증해 소독 \n - 재배사 내의 공기 중 습도는 90∼95%, 배지 내 의 수분 함량은 65∼70%가 되게 균상 관리하여 병 발생을 억제 \n - 한 주기가 끝나면 균상에 관수를 충분히 하고 버섯 발생 기에는 관수를 자제한다. \n - 균상에 관수한 후 즉시 환기를 실시하여 균상 표면 의 과다한 물방울이 증발되도록 하며, 재배사의 단열을 보완하여 밤낮의 온도편차 를 줄이고, 재배사 벽면, 버섯, 균상에 물방울이 생기지 않도록 한다.')
    
    elif pred_class == 'oyster_brown' :
        c2.write('느타리버섯 세균갈색무늬병 \n - 병원균을 전파하는 버섯파리, 응애의 방제철저. \n - 재배사의 바닥과 주위의 토양을 소독하며, 관수로 사용하는 지하수 저수조를 정기적으로 소독한다. \n - 버섯 생육에 알맞도록 환경을 조절하고 균상정리를 철저히 한다. \n - 폐상 퇴비의 살균처리를 철저히한다. \n - 재배사의 보온력을 높여 밤낮의 온도편차를 줄인다. \n - 안개가 심하게 발생하는 시기에는 저녁에 환기를 억제한다. \n - 갈색무늬병이 이미 발생한 경우 전체적으로 균상표면에 발생한 버섯을 제거한다. \n - 아그렙토, 브라마아신수화제를 3000배액으로 살포하여 균상표면의 병원균의 밀도를 낮춘다. 단, 방제약제 효과는 버섯생육에 알맞는 환경조건을 충족해야만 나타날 수 있다.')
    
    elif pred_class == 'pleurotus_brown':
        c2.write('큰느타리버섯 세균갈색무늬병 \n - 병원균을 전파하는 버섯파리, 응애의 방제철저. \n - 재배사의 바닥과 주위의 토양을 소독하며, 관수로 사용하는 지하수 저수조를 정기적으로 소독한다. \n - 버섯 생육에 알맞도록 환경을 조절하고 균상정리를 철저히 한다. \n - 폐상 퇴비의 살균처리를 철저히한다. \n - 재배사의 보온력을 높여 밤낮의 온도편차를 줄인다. \n - 안개가 심하게 발생하는 시기에는 저녁에 환기를 억제한다. \n - 갈색무늬병이 이미 발생한 경우 전체적으로 균상표면에 발생한 버섯을 제거한다. \n - 아그렙토, 브라마아신수화제를 3000배액으로 살포하여 균상표면의 병원균의 밀도를 낮춘다. 단, 방제약제 효과는 버섯생육에 알맞는 환경조건을 충족해야만 나타날 수 있다.')
    
    elif pred_class == 'portobello_brown' :
        c2.write('양송이버섯 세균갈색무늬병 \n - 병원균을 전파하는 버섯파리, 응애의 방제철저. \n - 재배사의 바닥과 주위의 토양을 소독하며, 관수로 사용하는 지하수 저수조를 정기적으로 소독한다. \n - 버섯 생육에 알맞도록 환경을 조절하고 균상정리를 철저히 한다. \n - 폐상 퇴비의 살균처리를 철저히한다. \n - 재배사의 보온력을 높여 밤낮의 온도편차를 줄인다. \n - 안개가 심하게 발생하는 시기에는 저녁에 환기를 억제한다. \n - 갈색무늬병이 이미 발생한 경우 전체적으로 균상표면에 발생한 버섯을 제거한다. \n - 아그렙토, 브라마아신수화제를 3000배액으로 살포하여 균상표면의 병원균의 밀도를 낮춘다. 단, 방제약제 효과는 버섯생육에 알맞는 환경조건을 충족해야만 나타날 수 있다.')
    
    elif pred_class == 'shiitake_brown':
        c2.write('표고버섯 세균갈색무늬병 \n - 병원균을 전파하는 버섯파리, 응애의 방제철저. \n - 재배사의 바닥과 주위의 토양을 소독하며, 관수로 사용하는 지하수 저수조를 정기적으로 소독한다. \n - 버섯 생육에 알맞도록 환경을 조절하고 균상정리를 철저히 한다. \n - 폐상 퇴비의 살균처리를 철저히한다. \n - 재배사의 보온력을 높여 밤낮의 온도편차를 줄인다. \n - 안개가 심하게 발생하는 시기에는 저녁에 환기를 억제한다. \n - 갈색무늬병이 이미 발생한 경우 전체적으로 균상표면에 발생한 버섯을 제거한다. \n - 아그렙토, 브라마아신수화제를 3000배액으로 살포하여 균상표면의 병원균의 밀도를 낮춘다. 단, 방제약제 효과는 버섯생육에 알맞는 환경조건을 충족해야만 나타날 수 있다.')
    
    elif pred_class == 'pleurotus_white' :
        c2.write('큰느타리버섯 흰곰팡이병 \n - 흰곰팡이병 발생이 확인된 경우 버섯 전체를 소금으로 덮어 포자의 비산을 막는다. \n - 발견 즉시 물에 젖은 종이타월을 오염된 부위를 조심스럽게 덮고 그위를 다시 소금으로 충분히 덮어준다. \n - 높은 습도와 19-22도의 온도조건에서 가장 많이 발생된다. \n - 가급적 빠른 시일 내에 오염된 배지를 배출하고 재배사 전체를 65-70도에서 5-6시간정도 스팀소독을한다. \n - 병징이 발견된 후 이 병균을 제거하는 것은 사실상 불가능하므로 예방이 최선이다. \n - 복토재료 소독 및 오염 방지 \n - 복토 후 0.1% 벤노밀 관수 \n - 수확 후 남는 버섯 대 제거 \n - 재배사 주변 소독과 청결상태 유지')
    
    elif pred_class == 'portobello_white' :
        c2.write('양송이버섯 흰곰팡이병 \n - 흰곰팡이병 발생이 확인된 경우 버섯 전체를 소금으로 덮어 포자의 비산을 막는다. \n - 발견 즉시 물에 젖은 종이타월을 오염된 부위를 조심스럽게 덮고 그위를 다시 소금으로 충분히 덮어준다. \n - 높은 습도와 19-22도의 온도조건에서 가장 많이 발생된다. \n - 가급적 빠른 시일 내에 오염된 배지를 배출하고 재배사 전체를 65-70도에서 5-6시간정도 스팀소독을한다. \n - 병징이 발견된 후 이 병균을 제거하는 것은 사실상 불가능하므로 예방이 최선이다. \n - 복토재료 소독 및 오염 방지 \n - 복토 후 0.1% 벤노밀 관수 \n - 수확 후 남는 버섯 대 제거 \n - 재배사 주변 소독과 청결상태 유지')    
    
    elif pred_class == 'shiitake_white' :
        c2.write('표고버섯 흰곰팡이병 \n - 흰곰팡이병 발생이 확인된 경우 버섯 전체를 소금으로 덮어 포자의 비산을 막는다. \n - 발견 즉시 물에 젖은 종이타월을 오염된 부위를 조심스럽게 덮고 그위를 다시 소금으로 충분히 덮어준다. \n - 높은 습도와 19-22도의 온도조건에서 가장 많이 발생된다. \n - 가급적 빠른 시일 내에 오염된 배지를 배출하고 재배사 전체를 65-70도에서 5-6시간정도 스팀소독을한다. \n - 병징이 발견된 후 이 병균을 제거하는 것은 사실상 불가능하므로 예방이 최선이다. \n - 복토재료 소독 및 오염 방지 \n - 복토 후 0.1% 벤노밀 관수 \n - 수확 후 남는 버섯 대 제거 \n - 재배사 주변 소독과 청결상태 유지')    
    
    elif pred_class == 'winter_white':
        c2.write('팽이버섯 흰곰팡이병 \n - 흰곰팡이병 발생이 확인된 경우 버섯 전체를 소금으로 덮어 포자의 비산을 막는다. \n - 발견 즉시 물에 젖은 종이타월을 오염된 부위를 조심스럽게 덮고 그위를 다시 소금으로 충분히 덮어준다. \n - 높은 습도와 19-22도의 온도조건에서 가장 많이 발생된다. \n - 가급적 빠른 시일 내에 오염된 배지를 배출하고 재배사 전체를 65-70도에서 5-6시간정도 스팀소독을한다. \n - 병징이 발견된 후 이 병균을 제거하는 것은 사실상 불가능하므로 예방이 최선이다. \n - 복토재료 소독 및 오염 방지 \n - 복토 후 0.1% 벤노밀 관수 \n - 수확 후 남는 버섯 대 제거 \n - 재배사 주변 소독과 청결상태 유지')        
    
st.write('Please upload image to be classified')




