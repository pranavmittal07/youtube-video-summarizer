from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
import json
import os
from pytubefix import YouTube
import assemblyai as aai
from openai import OpenAI
from .models import BlogPost
from dotenv import load_dotenv
from pytube.exceptions import PytubeError

load_dotenv()

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
    except PytubeError as e:
        print(f"Error fetching YouTube title: {e}")
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

def generate_blog_content(transcription):
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),  # this is also the default, it can be omitted
    )
    
    prompt = (
        "Write a detailed and well-structured Summary based on the following YouTube video transcript. "
        "Ensure it reads naturally as a blog, not as a video summary.\n\n"
        f"{transcription}\n\nArticle:"
    )

    try:
        response = client.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=1000
        )
        return response.choices[0].text.strip() if response.choices else None
    except Exception as e:
        print(f"Error generating blog content: {e}")
        return None

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
