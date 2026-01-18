import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Use sr.AudioFile(filename) as the audio source,
    recognize speech from that source,
    and return the recognized text.

    @params:
    filename (str) - path to the WAV audio file
    language (str) - language code (e.g., "en-US", "ja-JP")

    @returns:
    text (str) - the recognized speech as text
    '''
    
    # Create a recognizer object
    recognizer = sr.Recognizer()

    try:
        # Load the audio file
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)

        # Perform speech recognition using Google API
        text = recognizer.recognize_google(audio, language=language)
        return text

    except sr.UnknownValueError:
        # Speech was not clear
        return "Error: Speech could not be understood."

    except sr.RequestError:
        # API or internet issue
        return "Error: Could not connect to the speech recognition service."
