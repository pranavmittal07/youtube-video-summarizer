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
import assemblyai as aai
from .models import BlogPost
from dotenv import load_dotenv
import yt_dlp
from groq import Groq
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

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
    
    video_title = fetch_yt_title(yt_link)
    transcription = transcribe_audio(yt_link)
    
    if not transcription:
        return JsonResponse({'error': 'Failed to transcribe video'}, status=500)

    blog_content = generate_blog_content(transcription)

    if not blog_content:
        return JsonResponse({'error': 'Failed to generate blog content'}, status=500)
    
    BlogPost.objects.create(
        user=request.user,
        youtube_title=video_title,
        youtube_link=yt_link,
        generated_content=blog_content
    )
    
    return JsonResponse({'content': blog_content})

def fetch_yt_title(link):
    try:
        ydl_opts = {'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=False)
            return info.get('title', 'Unknown Title')
    except Exception as e:
        print(f"Error fetching YouTube title: {e}")
        return "Unknown Title"

def download_audio(link):
    try:
        output_template = os.path.join(settings.MEDIA_ROOT, '%(id)s.%(ext)s')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'postprocessors': [],  # No ffmpeg post-processing
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=True)
            file_ext = info.get('ext', 'webm')  # Usually webm
            file_path = os.path.join(settings.MEDIA_ROOT, f"{info['id']}.{file_ext}")
            return file_path
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
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    processed_words = [
        lemmatizer.lemmatize(stemmer.stem(word)) 
        for word in words if word not in stop_words and word.isalnum()
    ]

    return " ".join(processed_words)

def split_text(text, max_words=300):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def filter_with_tfidf(chunks):
    vectorizer = TfidfVectorizer()
    filtered_chunks = []

    for chunk in chunks:
        sentences = sent_tokenize(chunk)
        if not sentences:
            continue

        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores = tfidf_matrix.mean(axis=1).A1
        threshold = sorted(scores, reverse=True)[len(scores) // 2]
        important_sentences = [sentences[i] for i, score in enumerate(scores) if score >= threshold]

        filtered_chunks.append(" ".join(important_sentences))

    return filtered_chunks

def generate_blog_content(transcription):
    client = Groq(api_key=os.environ.get("GROQ"))
    processed_text = preprocess_text(transcription)
    chunks = split_text(processed_text)
    filtered_chunks = filter_with_tfidf(chunks)
    responses = []

    for i, chunk in enumerate(filtered_chunks):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        You are an advanced YouTube Video Summarizer.
                        Summarize the following content for a blog:
                        
                        Transcript:
                        {chunk}
                        """,
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            if chat_completion.choices:
                responses.append(chat_completion.choices[0].message.content)

            time.sleep(5)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            time.sleep(15)
            continue

    return " ".join(responses)

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
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('repeatPassword')

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
