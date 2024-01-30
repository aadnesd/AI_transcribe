import tempfile
import streamlit as st
import openai
import os
from pydub import AudioSegment
from pydub.utils import make_chunks, mediainfo
import math
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file 

temp_dir = tempfile.TemporaryDirectory()

#OPENAI Key and Org
# openai.api_key = os.environ['OPENAI_API_KEY'] #OpenAI API key.
openai.api_key = st
openai.organization = st.secrets['OPENAI_ORGANIZATION']

#Extract openai Model list 
#print(openai.Model.list())

#Setup streamlit page configurations, and file uploader for chosen file formats
st.set_page_config(page_title="Create video documentation!", page_icon=":robot:")
st.title("Video-transcription documenter")
uploaded_file = st.file_uploader("Upload a .wav or .mp4 file to transcribe and document:", type=[".wav", ".mp4"])
st.info(
                f"""
                        👆 Upload a .wav or .mp4 file. Or try a sample: [WAV sample](https://github.com/CharlyWargnier/CSVHub/blob/main/Wave_files_demos/Welcome.wav?raw=true) | [MP4 Sample](https://vimeo.com/665800536)
                        """
            )

#Check the file size (approximation), and provide error if file size is too large. 
if uploaded_file is not None:
    path_in = uploaded_file.name
    # Get file size from buffer
    # Source: https://stackoverflow.com/a/19079887
    old_file_position = uploaded_file.tell()
    uploaded_file.seek(0, os.SEEK_END)
    getsize = uploaded_file.tell()  # os.path.getsize(path_in)
    uploaded_file.seek(old_file_position, os.SEEK_SET)
    getsize = round((getsize / 1000000), 1)
    st.caption("The size of this file is: " + str(getsize) + "MB")

    if getsize < 100:  # File more than 2MB
        st.success("OK, less than 100 MB")
                
    else:
        st.error("More than 100 MB! Please use your own API")
        st.stop()
    

#Function for splitting uploaded file into chunks (based on file size), and saving to chunk-folder.
def video_chunkifyer_25(original_path):
    #If for å sjekke om det er mp4 eller wav
    if(((original_path.name).split('.')[-1].lower()) != 'wav'):
        myaudio = AudioSegment.from_file(original_path, "mp4")
    else:
        myaudio = AudioSegment.from_file(original_path , "wav")
    #channel_count = myaudio.channels    #Get channels
    #sample_width = myaudio.sample_width #Get sample width
    #duration_in_sec = len(myaudio) / 1000#Length of audio in sec
    #sample_rate = myaudio.frame_rate
    #print("sample_width=" + str(sample_width)) 
    #print("channel_count="+ str(channel_count))
    #print("duration_in_sec=" + str(duration_in_sec))
    #print("frame_rate=" + str(sample_rate))
    #bit_rate =16  #assumption , you can extract from mediainfo("test.wav") dynamically

    #wav_file_size = (sample_rate * bit_rate * channel_count * duration_in_sec) / 8
    #print("wav_file_size = " + str(wav_file_size))

    #file_split_size = 25000000  # 25Mb OR 25, 000, 000 bytes
    #total_chunks =  wav_file_size // file_split_size

    #Get chunk size by following method #There are more than one ofcourse
    #for  duration_in_sec (X) -->  wav_file_size (Y)
    #So   whats duration in sec  (K) --> for file size of 10Mb
    #  K = X * 10Mb / Y

    #chunk_length_in_sec = math.ceil((duration_in_sec * 10000000 ) /wav_file_size)   #in sec
    #chunk_length_ms = chunk_length_in_sec * 1000
    chunks = make_chunks(myaudio, 20000)

    #Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print("exporting" + chunk_name)
        chunk.export(temp_dir.name + '/' + chunk_name, format="wav")


# Function for transcribing audio file using openai Whisper
def transcribe_audio(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
    return transcription['text']


# Function for generating zendesk documentation from input transcription text (using davinci)
def generate_zendesk_documentation(transcription):
    prompt = f"""Translate the following transcription to Zendesk documentation, and it must be in github flavoured markdown language, ready for use as a .md file. Also, format it correctly with special symbols:

Transcription:
{transcription}

Instructions:
"""

    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=3000,
        temperature=0.7,
        n=1,
        stop=None
    )

    return response.choices[0].text.strip()

#generate documentation using gpt 4
def generate_zdoc(transcription):
    prompt = f"""Summarize the following transcription of a b2c conversation, and include key takeaways. If a SALE/significant incident occured, outline that output sentence with BOLD text. The output MUST strictly be in github flavored markdown language, and you must strictly make sure you use special symbol representations like "\$" correctly to avoid LaTeX syntax, and only markdown language is the working syntax. Avoid latex errors at all costs!

Transcription:
{transcription}
"""

    response = openai.ChatCompletion.create(
        model='gpt-4-0613',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=6000
    )

    return response.choices[0]['message']['content']

output = ""
#Streamlit buttons and output fields, and button logic execution (using the defined functions in the script)
if st.button("Transcribe file!"):
    #Logic for generating documentation: 
    video_chunkifyer_25(uploaded_file)
    te = ""
    #directory = '/tmp'
    for filename in sorted(os.listdir(temp_dir.name)):
        f = os.path.join(temp_dir.name, filename)
        if os.path.isfile(f):
            print(f)    
        te += transcribe_audio(f)

    st.subheader('Raw transcription text:')
    st.text_area(label="", value=te, height=200)

    output = generate_zdoc(te)
    st.subheader('Summary preview:')
    #st.text_area(label="", value=output, height=400)
    st.markdown(output)

    temp_dir.cleanup()
    #delete chunk files
    #for filename in os.listdir(directory):
    #    f = os.path.join(directory, filename)
    #    if os.path.isfile(f):
    #        os.remove(f)



