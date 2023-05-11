

import streamlit as st 
import altair as alt
import plotly.express as px 
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import numpy as np

import pandas as pd 
import numpy as np 
from datetime import datetime

import joblib 
pipe_lr = joblib.load(open("emotion_classifier_pipe_lr.pkl","rb"))


# Track Utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table


def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😁", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
	st.title("Emotion Classifier App")
	menu = ["Home","Monitor","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	create_page_visited_table()
	create_emotionclf_table()
	if choice == "Home":
		add_page_visited_details("Home",datetime.now())
		st.subheader("Home-Emotion In Text")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2  = st.columns(2)

			# Apply Fxn Here
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)
			
			add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))



			with col2:
				st.success("Prediction Probability")
				
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)



	elif choice == "Monitor":
		add_page_visited_details("Monitor",datetime.now())
		st.subheader("Monitor App")

		with st.beta_expander("Page Metrics"):
			page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
			st.dataframe(page_visited_details)	

			pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
			c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
			st.altair_chart(c,use_container_width=True)	

			p = px.pie(pg_count,values='Counts',names='Pagename')
			st.plotly_chart(p,use_container_width=True)

		with st.expander('Emotion Classifier Metrics'):
			df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
			st.dataframe(df_emotions)

			prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
			pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
			st.altair_chart(pc,use_container_width=True)	



	else:
		st.subheader("About")
		add_page_visited_details("About",datetime.now())
	st.sidebar.markdown("🛫We can analyse passengers review from this application.🛫")
#loading the data (the csv file is in the same folder)
#if the file is stored the copy the path and paste in read_csv method.
data=pd.read_csv('tweet.csv')
#checkbox to show data 
if st.checkbox("Show Data"):
    st.write(data.head(50))
#subheader
st.sidebar.subheader('Tweets Analyser')
#radio buttons
tweets=st.sidebar.radio('Sentiment Type',('positive','negative','neutral'))
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])

select=st.sidebar.selectbox('Visualisation Of Tweets',['Histogram','Pie Chart'],key=1)
sentiment=data['airline_sentiment'].value_counts()
sentiment=pd.DataFrame({'Sentiment':sentiment.index,'Tweets':sentiment.values})
st.markdown("###  Sentiment count")
if select == "Histogram":
        fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
        st.plotly_chart(fig)
else:
        fig = px.pie(sentiment, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

#slider
st.sidebar.markdown('Time & Location of tweets')
hr = st.sidebar.slider("Hour of the day", 0, 23)
data['Date'] = pd.to_datetime(data['tweet_created'])
hr_data = data[data['Date'].dt.hour == hr]
if not st.sidebar.checkbox("Hide", True, key='44'):
    st.markdown("### Location of the tweets based on the hour of the day")
    st.markdown("%i tweets during  %i:00 and %i:00" % (len(hr_data), hr, (hr+1)%24))
    st.map(hr_data)


st.sidebar.subheader("Airline tweets by sentiment")
choice = st.sidebar.multiselect("Airlines", ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key = '0')  
if len(choice)>0:
    air_data=data[data.airline.isin(choice)]
    
    fig1 = px.histogram(air_data, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment',labels={'airline_sentiment':'tweets'}, height=600, width=800)
    st.plotly_chart(fig1)




if __name__ == '__main__':
	main()
