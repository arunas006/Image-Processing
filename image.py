import numpy as np

from PIL import Image
pip install opencv-python
import pandas as pd
import cv2
import streamlit as st
from sklearn.decomposition import PCA


st.set_page_config(layout='wide')

st.write("""
    
    # **IMAGE PROCESSING TOOL**
    

    """
)

file = st.file_uploader("Choose the image which needs to be processed", accept_multiple_files=False,type=['jpg','png'])

if file is not None:
    image = Image.open(file)



    img = np.array(image)





    fea=0
    def image_red(img):
        
        blue,green,red=cv2.split(img)
        blue_mode=pd.DataFrame(data=blue/255)
        green_mode=pd.DataFrame(data=green/255)
        red_mode=pd.DataFrame(data=red/255)
        pca_blue=PCA()
        blue_pca=pca_blue.fit_transform(blue_mode)
        var=pca_blue.explained_variance_ratio_.cumsum()
        comp=len(var[var<=0.96])

        global fea

        fea=comp

        pca_blue=PCA(n_components=comp)
        blue_pca=pca_blue.fit_transform(blue_mode)

        pca_green=PCA(n_components=comp)
        green_pca=pca_green.fit_transform(green_mode)

        pca_red=PCA(n_components=comp)
        red_pca=pca_red.fit_transform(red_mode)

        blue_arr=pca_blue.inverse_transform(blue_pca)

        green_arr=pca_green.inverse_transform(green_pca)

        red_arr=pca_red.inverse_transform(red_pca)
        red_img=(cv2.merge((blue_arr,green_arr,red_arr)))

        st.image(red_img,use_column_width=True,clamp = True)
        



    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        
        # Actual Image
        

        """
        )
        st.text("Actual Shape of the image {}".format(img.shape[:2]))
        st.image(file)

    with col2:
        if(st.button("Convert")):
            st.write("""
        
            # Reduced Image

            """
            )
            image_red(img)
            st.text("Reduced Shape of the image {}".format((img.shape[0],fea)))
        
else:
    st.text("Please Upload the Image")  

