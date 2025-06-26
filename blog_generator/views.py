from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
import json
import os
import time
from pytubefix import YouTube
import assemblyai as aai
from openai import OpenAI
from .models import BlogPost
from dotenv import load_dotenv
from pytubefix.exceptions import VideoUnavailable
from groq import Groq
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

# Home Page (Protected)
@login_required
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def generate_blog(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
    try:
        data = json.loads(request.body)
        yt_link = data.get('link')
        if not yt_link:
            return JsonResponse({'error': 'YouTube link is required'}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    
    # Extract YouTube Title
    video_title = fetch_yt_title(yt_link)
    
    # Get Video Transcription
    transcription = transcribe_audio(yt_link)
    if not transcription:
        return JsonResponse({'error': 'Failed to get transcript'}, status=500)
    
    # Generate Blog Content
    blog_content = generate_blog_content(transcription)
    if not blog_content:
        return JsonResponse({'error': 'Failed to generate blog article'}, status=500)
    
    # Save Blog to Database
    blog_post = BlogPost.objects.create(
        user=request.user,
        youtube_title=video_title,
        youtube_link=yt_link,
        generated_content=blog_content
    )
    
    return JsonResponse({'content': blog_content})

def fetch_yt_title(link):
    try:
        yt = YouTube(link)
        return yt.title
    except VideoUnavailable as e:
        print(f"Error fetching YouTube title: {e}")
        return "Unknown Title"
    except Exception as e:
        print(f"General error fetching YouTube title: {e}")
        return "Unknown Title"

def download_audio(link):
    try:
        yt = YouTube(link)
        audio_stream = yt.streams.filter(only_audio=True).first()

        if not audio_stream:
            print("No audio stream found!")
            return None

        # Define file paths
        base_filename = yt.video_id  # Unique filename using video ID
        mp3_path = os.path.join(settings.MEDIA_ROOT, f"{base_filename}.mp3")

        # ✅ Check if the MP3 file already exists
        if os.path.exists(mp3_path):
            print(f"Audio file already exists: {mp3_path}")
            return mp3_path

        # Download the audio
        file_path = audio_stream.download(output_path=settings.MEDIA_ROOT)

        # Convert to MP3 (only if the .mp3 file doesn't already exist)
        new_mp3_path = os.path.splitext(file_path)[0] + '.mp3'

        if not os.path.exists(new_mp3_path):  # ✅ Prevent overwriting
            os.rename(file_path, new_mp3_path)

        return new_mp3_path
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def transcribe_audio(link):
    audio_file = download_audio(link)
    if not audio_file:
        return None

    aai.settings.api_key = os.getenv('ASSEMBLY_AI_API_KEY')

    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file)
        return transcript.text if transcript else None
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def preprocess_text(text):
    """Cleans and processes text: lowercasing, stopword removal, stemming, and lemmatization."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)  # Tokenize words
    processed_words = [
        lemmatizer.lemmatize(stemmer.stem(word)) 
        for word in words if word not in stop_words and word.isalnum()
    ]

    return " ".join(processed_words)  # Return cleaned text

def split_text(text, max_words=300):
    """Splits text into chunks of approximately max_words words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def filter_with_tfidf(chunks):
    """Ranks sentences using TF-IDF and filters out low-importance ones."""
    vectorizer = TfidfVectorizer()
    filtered_chunks = []

    for chunk in chunks:
        sentences = sent_tokenize(chunk)  # Split chunk into sentences
        tfidf_matrix = vectorizer.fit_transform(sentences)  # Compute TF-IDF scores
        scores = tfidf_matrix.mean(axis=1).A1  # Average score per sentence
        
        threshold = sorted(scores, reverse=True)[len(scores) // 2]  # Top 50% sentences
        important_sentences = [sentences[i] for i, score in enumerate(scores) if score >= threshold]
        
        filtered_chunks.append(" ".join(important_sentences))  # Combine important parts

    return filtered_chunks

def generate_blog_content(transcription):
    client = Groq(api_key=os.environ.get("GROQ"))  # Ensure API key is correctly set

    processed_text = preprocess_text(transcription)  # Preprocess transcript
    chunks = split_text(processed_text)  # Split into smaller chunks
    filtered_chunks = filter_with_tfidf(chunks)  # Remove unimportant parts
    responses = []

    for i, chunk in enumerate(filtered_chunks):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                            You are an advanced YouTube Video Summarizer. Your task is to create a structured, engaging, and information-rich summary based on the transcript of a YouTube video. The summary should condense key insights while ensuring clarity and readability.
                            
                            **Transcript**:
                            {chunk}

                            Summary Format:
                            Introduction:

                            What is this video about? (Summarize the core topic in 2-3 lines.)

                            Who is the speaker/creator, and what is their expertise (if known)?

                            Who is this video for? (Beginners, professionals, general audience?)

                            What is the main goal of the video? (To educate, review, entertain, inspire, compare, analyze, etc.)

                            Prerequisites (If Any):

                            Any background knowledge needed before watching?

                            Key concepts the speaker assumes the audience already knows.

                            Key Takeaways & Highlights:

                            Summarize the most important points covered in the video.

                            Break down key sections logically.

                            Use bullet points for clarity.

                            Include real-world applications or examples mentioned in the video.

                            Maintain a natural flow, ensuring it feels like a coherent summary rather than just a list of points.

                            If it's a review/comparison, highlight pros, cons, and verdict.

                            Actionable Insights (If Applicable):

                            Any steps, recommendations, or advice provided by the speaker?

                            What should the viewer do next after watching this video? (Further reading, next steps, practical applications, etc.)

                            Mind Map Format (For Visual Representation):

                            Provide a structured, parseable format that can be used to generate a mind map.

                            Use nested bullet points, JSON, or Markdown for hierarchical structuring.

                            Conclusion (If Relevant):

                            A final summary of the videos main message.

                            Any closing remarks from the speaker.

                            If the video is motivational, summarize the core inspirational takeaway.

                            Additional Guidelines:
                            Ensure the summary is concise yet detailed, allowing a 5-10 minute readable format.

                            Use bold text for emphasis, bullet points for clarity, and paragraph breaks for readability.

                            Remove filler content, redundant information, or off-topic discussions.

                            If the video includes recaps, summarize them briefly without unnecessary repetition.

                            For podcasts/interviews, highlight key insights or quotes from the guest(s).

                            The output should allow a reader to fully grasp the core content of the video without needing to watch it.

                            """,
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            if chat_completion.choices:
                responses.append(chat_completion.choices[0].message.content)
            
            time.sleep(5)  # Delay to prevent rate limits

        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            time.sleep(15)  # Longer wait before retrying
            continue  # Move to the next chunk

    return " ".join(responses)  # Merge all summarized chunks into one

@login_required
def blog_list(request):
    blogs = BlogPost.objects.filter(user=request.user)
    return render(request, 'all-blogs.html', {'blog_articles': blogs})

@login_required
def blog_details(request, pk):
    blog = get_object_or_404(BlogPost, id=pk, user=request.user)
    return render(request, 'blog-details.html', {'blog_article_detail': blog})

def user_login(request):
    if request.method == 'POST':
        username, password = request.POST.get('username'), request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user:
            login(request, user)
            return redirect('/')
        else:
            return render(request, 'login.html', {'error_message': 'Invalid username or password'})
    
    return render(request, 'login.html')

def user_signup(request):
    if request.method == 'POST':
        username, email = request.POST.get('username'), request.POST.get('email')
        password, confirm_password = request.POST.get('password'), request.POST.get('repeatPassword')
        
        if password != confirm_password:
            return render(request, 'signup.html', {'error_message': 'Passwords do not match'})
        
        if User.objects.filter(username=username).exists():
            return render(request, 'signup.html', {'error_message': 'Username already taken'})
        
        user = User.objects.create_user(username, email, password)
        login(request, user)
        return redirect('/')
    
    return render(request, 'signup.html')

def user_logout(request):
    logout(request)
    return redirect('/')
