import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain import LLMChain, PromptTemplate

# Load environment variables
load_dotenv()

# Define default tweets
DEFAULT_TWEETS = [
    "Mf*<krs always find $ to invest when the market is \"UP ONLY\" & seemingly easy money is on the table!",
    "I 👁️ a lot more people waking up to $TAO",
    "$TAO broke resistance of the downwards trend line.",
    "You have 2 CHOICES: 1. Position yourself before the PUMPS = WIN 2. Chase the PUMPS = LOSE",
    "An easy $100M A.I. agent infra project sitting at $25M & you're messing about with single agents?",
    "$ETH & the gas fees — sort it out",
    "LFG $BTC We did $100k! Currently $103k - it was inevitable.",
    "It's ONLY EASY NOW because you took the hard route previously.",
    "AI - AI Agents are the play.",
    "ALL INVESTMENTS are just vehicles to wealth."
]

# Define default LinkedIn posts
DEFAULT_LINKEDIN_POSTS = [
    "Legendary SongGPT Ai music platform. Mind blowing how good it is. Excited about the future of Ai content creation to create bloomscrolling. Thanks for having me :)  Youtube video of the talk here.",
    "On Persist's model of building a bridge to dark talent (founder potential) and the ecosystem of startups I've brought together. Enjoyed speaking about meme coins, shared belief systems, and the future of finance with the man Brock Pierce wearing our Meme Lords shirts, our stealth venture.",
    "We all know legacy media -> creator economy -> attention meritocracy And it will become more and more common knowledge that we are going through the legacy finance -> token economy -> shared belief system meritocracy",
    "The rate at which persistent people can learn, and create a niche app is unlike ever before. I've hired over 60 people this past year without looking at a single resume or doing an interview. We simply give a challenge to build a full MVP of the startup we are looking to create and let thousands of people compete for salary + equity!",
    "I first applied to Thiel Fellowship with Jim O'Neill in 2017, and to be on the other side, helping to find undiscovered talent is incredible. Jim is an amazing human and bouncing board for ideas and direction."
]

def generate_social_posts(post_type):
    """Generate tweets or LinkedIn posts using Groq LLM"""
    try:
        # Retrieve API key from environment variable
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found in environment variables")

        # Initialize the Groq LLM
        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=1,
            max_tokens=1024,
            top_p=1,
            groq_api_key=api_key
        )

        # Create different prompt templates for tweets and LinkedIn posts
        if post_type == "Tweets":
            template = """You are a Twitter influencer like Jackson (Jack Jay) Jesionowski (https://www.linkedin.com/in/jack-jay-jackson-jesionowski/), focused on tech, crypto, and entrepreneurship. 
            Based on these example tweets, generate 10 new tweets that capture the style, tone, and themes:

            Example Tweets:
            {example_posts}

            Guidelines for generated tweets:
            - Use concise, impactful language
            - Include thought-provoking insights
            - Mix tech, crypto, and entrepreneurship themes
            - Use minimal emojis
            - Maintain an authentic, visionary tone
            - Limit to 280 characters
            Make sure the cosine similary of the generated tweets are above 8
            Generate 10 unique tweets. 
            Format each tweet on a new line, starting with "Tweet X:":"""
            max_length = 280
            example_posts = "\n".join(DEFAULT_TWEETS)
        else:  # LinkedIn Posts
            template = """You are a content creator like Jackson (Jack Jay) Jesionowski, focused on tech entrepreneurship, startup culture, and innovation. 
            Based on these example LinkedIn posts, generate 10 new posts that capture the style, tone, and themes:

            Example LinkedIn Posts:
            {example_posts}

            Guidelines for generated LinkedIn posts:
            - Use inspirational and visionary language
            - Share insights about technology, entrepreneurship, and personal growth
            - Include networking and professional development themes
            - Maintain an authentic, motivational tone
            - Aim for 1-3 paragraphs
            Make sure the cosine similary of the generated tweets are above 8
            Generate 10 unique LinkedIn posts. 
            Format each post on a new line, starting with "Post X:":"""
            max_length = 3000
            example_posts = "\n".join(DEFAULT_LINKEDIN_POSTS)

        # Create prompt with post examples
        prompt = PromptTemplate(
            input_variables=["example_posts"],
            template=template,
        )

        # Create an LLMChain for post generation
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Generate posts
        response = llm_chain.run(example_posts=example_posts)

        # Extract posts using regex
        post_pattern = r'(?:Tweet|Post) \d+:\s*(.+?)(?=(?:Tweet|Post) \d+:|$)'
        generated_posts = re.findall(post_pattern, response, re.DOTALL)

        # Clean and truncate posts
        generated_posts = [
            post.strip()[:max_length] for post in generated_posts 
            if post.strip()
        ][:10]

        return generated_posts

    except Exception as e:
        st.error(f"Error in generate_social_posts: {str(e)}")
        return []

def main():
    # Set page config
    st.set_page_config(page_title="Social Media Post Generator", page_icon="🚀", layout="centered")

    # Title and description
    st.title("🚀 Social Media Post Generator")
    st.write("Generate tweets and LinkedIn posts inspired by tech visionaries")

    # Validate Groq API Key exists
    if not os.getenv("GROQ_API_KEY"):
        st.error("Groq API key not found in environment variables. Please set GROQ_API_KEY.")
        return

    # Select post type
    post_type = st.selectbox("Select Post Type", ["Tweets", "LinkedIn Posts"])

    # Generate button
    if st.button(f"Generate {post_type}"):
        with st.spinner(f'Generating {post_type.lower()}...'):
            try:
                # Generate posts
                generated_posts = generate_social_posts(post_type)

                # Display generated posts
                if generated_posts:
                    st.subheader(f"Generated {post_type}:")
                    for i, post in enumerate(generated_posts, 1):
                        st.markdown(f"**{post_type[:-1]} {i}:** {post}")
                else:
                    st.warning(f"No {post_type.lower()} were generated.")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Sidebar info
    st.sidebar.title("How to Use")
    st.sidebar.info(
        "1. Ensure GROQ_API_KEY is set in your environment\n"
        "2. Select post type (Tweets or LinkedIn Posts)\n"
        "3. Click 'Generate' to create posts\n"
        "4. View generated posts in the app"
    )

if __name__ == "__main__":
    main()