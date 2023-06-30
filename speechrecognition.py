import speech_recognition
import pyttsx3
def machinespeaks(command):
    voice = pyttsx3.init()
    voice.say(command)
    voice.runAndWait()


sr = speech_recognition.Recognizer()

with speech_recognition.Microphone() as mic:
    print(" Silence please")
    sr.adjust_for_ambient_noise(mic, duration=2)
    print(" Kindly Speak now....")

    audio = sr.listen(mic)

    text = sr.recognize_google(audio)

    text = text.lower()

    print("You said: " + text)

    machinespeaks(text)