import os
import json
import requests
import google.generativeai as genai
from datetime import datetime, timedelta

# -------------------------
# CONFIG
# -------------------------

# 🔑 API Keys
NEWS_API_KEY = "ca1502e590ea4a28b4ec37dcadf42188"
GEMINI_API_KEY = "AIzaSyDmZtInmhGnwYODL-3pd0VgMuxGiDBIi6c"

# JSON file path (you can change this)
JSON_FILE_PATH = "data.json"  # Replace with your JSON file path

# -------------------------
# FUNCTIONS
# -------------------------

def analyze_json_file(file_path):
    """Analyze JSON file and extract relevant keywords using Gemini"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Convert JSON to string for analysis
        json_text = json.dumps(json_data, indent=2)
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        prompt = f"""
        Analyze the following JSON data and extract exactly 5 relevant keywords or topics that would be good for news searching.
        
        Requirements:
        1. Return only 5 keywords/phrases
        2. Each keyword should be 1-3 words maximum
        3. Focus on technology, business, science, or current affairs topics
        4. Make keywords suitable for news searches
        5. Return as a simple comma-separated list with no other text
        
        JSON Data:
        {json_text}
        
        Return format: keyword1, keyword2, keyword3, keyword4, keyword5
        """
        
        response = model.generate_content(prompt)
        keywords_text = response.text.strip()
        
        # Parse the keywords
        keywords = [k.strip() for k in keywords_text.split(',')]
        
        # Ensure we have exactly 5 keywords
        keywords = keywords[:5] if len(keywords) >= 5 else keywords
        
        return keywords
        
    except Exception as e:
        print(f"Error analyzing JSON file: {e}")
        # Fallback keywords if JSON analysis fails
        return ["artificial intelligence", "technology", "business", "innovation", "market trends"]


def fetch_news_articles(query, num=5):
    """Fetch recent news articles from NewsAPI"""
    try:
        # Get articles from the last 7 days
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': num,
            'from': from_date
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching news for {query}: {e}")
        return None


def prepare_data(topics):
    """Collect and format news data from dynamic topics"""
    print("Fetching latest news articles...")
    all_articles = []
    
    for topic in topics:
        print(f"Searching for: {topic}")
        news_data = fetch_news_articles(topic)
        
        if news_data and news_data.get('status') == 'ok':
            articles = news_data.get('articles', [])
            
            for article in articles:
                title = article.get('title', 'No title')
                description = article.get('description', 'No description')
                url = article.get('url', 'No URL')
                published_at = article.get('publishedAt', 'Unknown date')
                source = article.get('source', {}).get('name', 'Unknown source')
                
                # Clean up the data
                if title != 'No title' and description != 'No description':
                    article_info = {
                        'topic': topic,
                        'title': title,
                        'description': description,
                        'url': url,
                        'published_at': published_at,
                        'source': source
                    }
                    all_articles.append(article_info)
    
    return all_articles


def format_articles_for_gemini(articles):
    """Format articles into a structured text for Gemini analysis"""
    if not articles:
        return "No articles found."
    
    formatted_text = "Recent news articles for analysis:\n\n"
    
    for i, article in enumerate(articles, 1):
        formatted_text += f"Article {i}:\n"
        formatted_text += f"Topic: {article['topic'].title()}\n"
        formatted_text += f"Title: {article['title']}\n"
        formatted_text += f"Description: {article['description']}\n"
        formatted_text += f"Source: {article['source']}\n"
        formatted_text += f"Published: {article['published_at']}\n"
        formatted_text += f"URL: {article['url']}\n"
        formatted_text += "-" * 80 + "\n\n"
    
    return formatted_text


def analyze_with_gemini(articles_text, topics):
    """Send collected articles to Gemini for comprehensive summarization"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        topics_list = ", ".join(topics)
        
        prompt = f"""
        Please analyze the following recent news articles and create a comprehensive summary report.

        Instructions:
        1. Group the articles by these topics: {topics_list}
        2. For each topic, provide a summary paragraph highlighting the key developments and trends
        3. Include important details, statistics, and implications mentioned in the articles
        4. Write in a clear, professional news summary format using plain text only
        5. Do not use any markdown formatting, asterisks, or special characters for emphasis
        6. Include relevant source citations where appropriate
        7. End with a brief overall outlook section
        8. Use proper paragraph breaks and clear section headings

        Here are the articles:

        {articles_text}
        
        Please provide a well-structured summary in clean paragraph format without any formatting symbols.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Error analyzing with Gemini:", e)
        return None


def clean_formatting(text):
    """Remove markdown formatting and clean up text"""
    if not text:
        return text
    
    # Remove common markdown formatting
    text = text.replace('**', '')
    text = text.replace('*', '')
    text = text.replace('__', '')
    text = text.replace('_', '')
    text = text.replace('##', '')
    text = text.replace('#', '')
    
    # Clean up multiple spaces and line breaks
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    return '\n\n'.join(cleaned_lines)


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    print("🚀 Starting Dynamic News Summary Generator...")
    
    # Step 1: Analyze JSON file to get keywords
    print("📊 Analyzing JSON file for relevant keywords...")
    
    if os.path.exists(JSON_FILE_PATH):
        keywords = analyze_json_file(JSON_FILE_PATH)
        print(f"✅ Extracted keywords: {', '.join(keywords)}")
    else:
        print(f"⚠️  JSON file not found at {JSON_FILE_PATH}. Using default keywords.")
        keywords = ["artificial intelligence", "technology", "business", "innovation", "market trends"]
    
    # Step 2: Collect news articles using dynamic keywords
    articles = prepare_data(keywords)
    
    if not articles:
        print("❌ No articles collected. Exiting...")
        exit()
    
    print(f"✅ Collected {len(articles)} articles")
    
    # Step 3: Format articles for Gemini
    formatted_articles = format_articles_for_gemini(articles)
    
    print("🤖 Analyzing articles with Gemini AI...")
    
    # Step 4: Get AI summary
    summary = analyze_with_gemini(formatted_articles, keywords)
    
    if summary:
        # Clean the summary of formatting
        cleaned_summary = clean_formatting(summary)
        
        print("\n" + "="*60)
        print("📰 DAILY NEWS SUMMARY")
        print("="*60)
        print(f"Keywords analyzed: {', '.join(keywords)}")
        print("="*60 + "\n")
        print(cleaned_summary)
        
        # Save summary to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dynamic_news_summary_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Dynamic News Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n")
            f.write(f"Keywords extracted from JSON: {', '.join(keywords)}\n")
            f.write("="*60 + "\n\n")
            f.write(cleaned_summary)
            f.write("\n\n" + "="*60 + "\n")
            f.write("Raw Articles Data:\n\n")
            f.write(formatted_articles)
        
        print(f"\n✅ Summary saved to {filename}")
    else:
        print("❌ No summary generated.")
    
    print("\n🎉 Process completed!")