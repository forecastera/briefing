"""
Audio Briefing Generation API
-----------------------------
A production-ready FastAPI service that processes documents from S3 buckets,
generates audio briefings using OpenAI's GPT-4 and TTS models, and stores
results in another S3 bucket.

Key Naming Convention:
- Input files must follow strict naming format:
  [Account]_[ReportType]_For_[Client]_[Any_Suffix].{pdf|csv|txt}
  - Account: Organization name (underscore-separated words)
  - Client: Recipient name (underscore-separated words)
  - ReportType: Must be "Intelligence_Report" or "ABM_Report"
  - Suffix: Any additional identifiers (ignored in processing)
  Examples:
  - Global_Camping_Suppliers_Intelligence_Report_For_National_Outdoor_Retailers_2025.pdf
  - Regional_Distributors_ABM_Report_For_Acme_Corp_LLC_Final.csv

Environment Variables Required:
AWS_ACCESS_KEY_ID:       AWS IAM access key with S3 permissions
AWS_SECRET_ACCESS_KEY:   AWS IAM secret key
OPENAI_API_KEY:          OpenAI API key for GPT/TTS access
S3_SOURCE_BUCKET:        S3 bucket name for input documents
S3_DEST_BUCKET:          S3 bucket name for output audio files
AWS_REGION:              AWS region (default: us-east-1)
SSL_KEYFILE:             [Optional] Path to SSL key file
SSL_CERTFILE:            [Optional] Path to SSL certificate file

Execution Instructions:
1. Install dependencies:
   pip install fastapi uvicorn boto3 openai pydub pandas pypdf2 tiktoken python-dotenv
2. Create .env file with required credentials
3. Start the service:
   uvicorn main:app --host 0.0.0.0 --port 8000
4. Access API documentation at http://localhost:8000/docs
"""

import boto3
import os
import io
import logging
import uuid
import re
from typing import Dict, Optional
from openai import OpenAI
import tiktoken
import PyPDF2
import pandas as pd
from pydub import AudioSegment
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import uvicorn

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Audio Briefing API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

class ProcessingRequest(BaseModel):
    """
    API request model for processing documents
    
    Parameters:
    - s3_key: Full S3 path to input document following naming convention:
      [Account]_[ReportType]_For_[Client]_[Any_Suffix].{pdf|csv|txt}
    - report_type: Type of report (default: intelligence_report)
    """
    s3_key: str
    report_type: str = "intelligence_report"

class BriefingGenerator:
    """
    Core document processing and audio generation engine
    
    Configuration Parameters:
    - openai_model: GPT model version (default: gpt-4o)
    - tts_model: Text-to-speech model (default: tts-1-hd)
    - tts_voice: Voice style for TTS (default: ash)
    - max_retries: Number of retry attempts for S3 operations
    - chunk_size: Text processing chunk size for large documents
    """
    
    def __init__(self, config: Dict):
        """Initialize service with AWS and OpenAI configurations"""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config['aws_access_key_id'],
            aws_secret_access_key=config['aws_secret_access_key'],
            region_name=config.get('aws_region', 'us-east-1')
        )
        self.source_bucket = config['source_bucket']
        self.destination_bucket = config['destination_bucket']
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
    def process_file(self, s3_key: str) -> Dict[str, str]:
        """
        Main document processing pipeline
        Args:
            s3_key: S3 object key for input document
        Returns:
            Dictionary with processing status and audio file URLs
        """
        try:
            # File retrieval and processing
            file_bytes, filename = self._download_from_s3(s3_key)
            content, account_name, client_name, report_type = self._parse_file(file_bytes, filename)
            
            # Audio generation workflow
            audio_urls = self._generate_audio_files(
                content=content,
                account_name=account_name,
                client_name=client_name,
                report_type=report_type
            )
            
            return {
                "status": "success",
                "audio_files": audio_urls,
                "metadata": {
                    "account": account_name,
                    "client": client_name,
                    "report_type": report_type
                }
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _download_from_s3(self, s3_key: str) -> tuple:
        """
        Download file from source S3 bucket
        Args:
            s3_key: Full S3 path to document
        Returns:
            Tuple of (file_bytes, filename)
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.source_bucket,
                Key=s3_key
            )
            return response['Body'].read(), os.path.basename(s3_key)
        except ClientError as e:
            logger.error(f"S3 download error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid S3 path or permissions"
            )

    def _upload_to_s3(self, file_bytes: bytes, s3_key: str) -> str:
        """
        Upload generated audio file to destination S3 bucket
        Args:
            file_bytes: Audio bytes to upload
            s3_key: Destination S3 path
        Returns:
            S3 URI of uploaded file
        """
        try:
            self.s3_client.put_object(
                Bucket=self.destination_bucket,
                Key=s3_key,
                Body=file_bytes,
                ContentType='audio/mpeg'
            )
            return f"s3://{self.destination_bucket}/{s3_key}"
        except ClientError as e:
            logger.error(f"S3 upload error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upload audio file"
            )

    def _parse_file(self, file_bytes: bytes, filename: str) -> tuple:
        """
        Extract content and metadata from uploaded file
        Args:
            file_bytes: Raw file bytes
            filename: Original filename
        Returns:
            Tuple of (content, account_name, client_name, report_type)
        """
        account_name, client_name, report_type = self._extract_metadata(filename)
        
        # Process different file types
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            content = " ".join(page.extract_text() for page in pdf_reader.pages)
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
            content = df.to_string()
        else:  # TXT file
            content = file_bytes.decode("utf-8", errors="replace")
            
        return content, account_name, client_name, report_type

    def _extract_metadata(self, filename: str) -> tuple:
        """
        Extract account/client names and report type from filename
        Supported patterns:
        - [Account]_Intelligence_Report_For_[Client]_*
        - [Account]_ABM_Report_For_[Client]_*
        
        Example valid filenames:
        - Global_Camping_Suppliers_Intelligence_Report_For_National_Outdoor_Retailers_2025.pdf
        - Regional_Distributors_ABM_Report_For_Acme_Corp_LLC_Final.csv
        """
        filename = filename.lower()
        patterns = {
            "intelligence_report": r"^([a-z0-9_]+(?:_[a-z0-9_]+)*)_intelligence_report_for_([a-z0-9_]+(?:_[a-z0-9_]+)*)(?:_|\.|$)",
            "ABM_report": r"^([a-z0-9_]+(?:_[a-z0-9_]+)*)_abm_report_for_([a-z0-9_]+(?:_[a-z0-9_]+)*)(?:_|\.|$)"
        }
        
        for report_type, pattern in patterns.items():
            match = re.match(pattern, filename)
            if match:
                account = match.group(1).replace("_", " ").title()
                client = match.group(2).replace("_", " ").title()
                return account, client, report_type
                
        logger.error(f"Invalid filename format: {filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must follow format: [Account]_[ReportType]_For_[Client]_*"
        )

    def _generate_audio_files(self, content: str, account_name: str, 
                            client_name: str, report_type: str) -> Dict[str, str]:
        """
        Generate and upload audio files with proper S3 key formatting
        Args:
            content: Processed document content
            account_name: Formatted account name
            client_name: Formatted client name
            report_type: Report type
        Returns:
            Dictionary of audio file URLs
        """
        audio_urls = {}
        
        try:
            # Convert names to safe S3 format
            account_safe = account_name.replace(" ", "_")
            client_safe = client_name.replace(" ", "_")
            
            if report_type == "intelligence_report":
                scripts = {
                    'exec': self._generate_intelligence_script(content, 'exec', account_name, client_name),
                    'businessoverview': self._generate_intelligence_script(content, 'businessoverview', account_name, client_name),
                    'competitors': self._generate_intelligence_script(content, 'competitors', account_name, client_name),
                    'stakeholders': self._generate_intelligence_script(content, 'stakeholders', account_name, client_name)
                }
            elif report_type == "ABM_report":
                scripts = {'abm': self._generate_abm_script(content, account_name, client_name)}
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
                
            for briefing_type, script in scripts.items():
                audio_bytes = self._text_to_speech(script)
                if audio_bytes:
                    s3_key = f"{account_safe}/{client_safe}/{briefing_type}_briefing_{uuid.uuid4()}.mp3"
                    audio_urls[briefing_type] = self._upload_to_s3(audio_bytes, s3_key)
                    
            return audio_urls
        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Audio generation pipeline failed"
            )

    def _generate_intelligence_script(self, content: str, section: str, 
                                    account: str, client: str) -> str:
        """
        Generate intelligence report script for specific section
        Args:
            content: Document content
            section: Briefing section type
            account: Account name
            client: Client name
        """
        prompt = self._create_intelligence_prompt(content, section, account, client)
        return self._generate_script(prompt)

    def _generate_abm_script(self, content: str, account: str, client: str) -> str:
        """
        Generate ABM report script
        Args:
            content: Document content
            account: Account name
            client: Client name
        """
        prompt = self._create_abm_prompt(content, account, client)
        return self._generate_script(prompt)

    def _create_intelligence_prompt(self, content: str, section: str, 
                                  account: str, client: str) -> str:
        """
        Create GPT prompt for intelligence report sections
        Returns:
            Formatted prompt string
        """
        file_contents=content,
        account_name=account,
        client_name=client
        prompts = {
            'exec': f"""
Create a dynamic and engaging TED Talk script for "Next Quarter's Executive Briefing" featuring a single speaker. The talk should focus on strategic priorities and key initiatives for {account_name}, using the provided content as the foundation:

{file_contents}

Follow these detailed guidelines to ensure the talk is conversational, insightful, and actionable:

1. Introduction:
   - Start exactly with "Welcome to Next Quarter's briefing on {account_name}, prepared exclusively for {client_name}."
   - Provide an outline of what will be covered in the talk to help listeners follow along.
   - Set the tone and start the introduction with an engaging opening statement.

2. First 4 Initiatives:
   - Count the total number of initiatives mentioned in the content and state: "Out of X initiatives, we are focusing on the top 4."
   - For each of these first four initiatives:
     - Summarize its context or importance to the client in an engaging way.
     - Discuss recommended alignment with practical, actionable suggestions.
     - Highlight one relevant case study that demonstrates success and aligns with the initiative.
     - Keep each initiative discussion concise while ensuring clarity and depth.
   - Present each initiative in a natural flow without alternating speakers.

3. Other Sections:
   - Summarize content from other sections (apart from initiatives) from a sales executive's perspective.
   - Focus on actionable insights or opportunities for selling into the account.
   - Use storytelling techniques to naturally connect ideas, incorporating relevant sales and marketing buzzwords.
   - Present these sections without alternating speakers.

4. Tone and Delivery:
   - Maintain an enthusiastic and conversational tone throughout.
   - Maitain a natural pace.
   - Avoid overly long pauses and don't include forced transitions like "oh great" or "absolutely."
   - Allow for natural reactions but avoid making them feel scripted or excessive.
   - Keep transitions smooth by leading entire sections without frequent back-and-forth exchanges.

5. Structure:
   - Conclude with a summary of key takeaways and a motivational call-to-action for listeners to drive engagement.
   - End with: "You can always find more details about {account_name} in the full intelligence report provided by Next Quarter."

6. Dialogue Style:
   - Focus on natural conversational phrasing without naming the speaker.
   - Encourage lighthearted yet professional dialogue when appropriate.

7. TED Talk-Style Approach:
   - Structure the talk like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
   - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

By following these guidelines, craft a compelling TED Talk script that informs, motivates, and captivates listeners while addressing strategic priorities effectively.

Script:
""",
            'businessoverview': f"""
Continue the "Next Quarter's Executive Briefing" TED Talk, focusing on the Business Overview and SWOT analysis based on the provided content:

{file_contents}

Follow these detailed guidelines to ensure a seamless, engaging, and insightful continuation of the talk:

1. Transition:
   - Begin with a smooth transition from the previous segment about initiatives.
   - Provide a brief outline of what will be covered in this segment (Business Overview, SWOT Analysis, Roadmap Commentary, Market and Industry Trends).

2. SWOT Analysis:
   - Highlight one key strength and one significant weakness from the SWOT analysis:
     - Provide real-world examples or context for each point to make them relatable.
     - Discuss their impact on business strategy or operations in practical terms.
   - Present both the strength and weakness in a natural flow without alternating speakers.

3. Roadmap Commentary:
   - Offer concise insights into the current and future roadmap:
     - Mention key milestones or goals that stand out.
     - Explain how the roadmap addresses both the highlighted strength and weakness effectively.
   - Present this section as part of the overall narrative.

4. Market Trend:
   - Identify one relevant market trend:
     - Discuss its potential impact on business strategy.
     - Suggest actionable ways the company could leverage this trend to its advantage.
   - Integrate this section smoothly into the talk.

5. Industry Trend:
   - Highlight one significant industry trend:
     - Analyze its implications for competitors and overall market dynamics.
     - Propose strategies for addressing this trend to maintain a competitive edge.
   - Present this section as a continuation of the market trend discussion.

6. Tone and Storytelling:
   - Use storytelling techniques to dynamically connect insights and ideas, keeping the dialogue natural and engaging.
   - Maintain an upbeat tone while focusing on smooth, natural phrasing.
   - Avoid overly long pauses or forced transitions like "oh great" or "absolutely."

7. Conclusion:
   - Summarize key points discussed in this segment concisely.
   - End with a teaser for the upcoming Competitors' section to keep listeners intrigued.

8. Dialogue Style:
   - Focus on natural conversational phrasing without naming the speaker.
   - Encourage lighthearted yet professional dialogue when appropriate.

9. TED Talk-Style Approach:
    - Structure this segment like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
    - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

By following these updated guidelines, craft a concise yet impactful TED Talk segment that informs, engages, and transitions smoothly into the next topic.

Script:
""",
            'competitors': f"""
Continue the "Next Quarter's Executive Briefing" TED Talk, shifting the focus to Competitors based on the provided content:

{file_contents}

Follow these detailed guidelines to ensure a seamless, engaging, and insightful continuation of the talk:

1. Transition:
   - Begin with a smooth transition from the Business Overview segment.
   - Provide a brief outline of what will be covered in this segment (Competitor Analysis, Client Gaps, Detailed Competitor Spotlight, and Countermeasures) to help listeners follow along.

2. Competitor Analysis:
   - Discuss key competitors:
     - Provide an overview of their strengths relative to the client.
     - Use specific examples to illustrate competitive dynamics and market positioning.
   - Present this section as part of the overall narrative.

3. Client Gaps:
   - Identify two areas where the client is lagging behind competitors:
     - Analyze the implications of these gaps on market position or sales performance.
     - Suggest actionable strategies to address these weaknesses effectively.
   - Integrate this section smoothly into the talk.

4. Detailed Competitor Spotlight:
   - Highlight one specific competitor in detail:
     - Discuss their unique sales tactics or strategies.
     - Analyze how these contribute to their success and differentiate them in the market.
   - Present this section as a continuation of the competitor analysis.

5. Countermeasures:
   - Present practical countermeasures for the client:
     - Suggest strategies that could bolster competitive positioning and mitigate risks effectively.
   - Integrate this section naturally into the narrative.

6. Tone and Engagement:
   - Maintain an engaging and conversational tone throughout, focusing on smooth, natural phrasing.
   - Avoid overly long pauses or forced transitions like "oh great" or "absolutely."
   - Use relevant buzzwords naturally within discussions to resonate with a professional audience.

7. Conclusion:
   - Summarize key points discussed in this segment concisely.
   - End with a teaser for the upcoming Stakeholders' section to keep listeners intrigued.

8. Dialogue Style:
   - Focus on natural conversational phrasing without naming the speaker.
   - Encourage lighthearted yet professional dialogue when appropriate.

9. TED Talk-Style Approach:
    - Structure this segment like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
    - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

By following these updated guidelines, craft a concise yet impactful TED Talk segment that informs, engages, and transitions smoothly into the next topic.

Script:
""",
            'stakeholders': f"""
Conclude the "Next Quarter's Executive Briefing" TED Talk by focusing on Key Stakeholders based on the provided content:

{file_contents}

Follow these detailed guidelines to craft a smooth, engaging, and impactful conclusion to the talk:

1. Transition:
   - Begin with a seamless and engaging transition from the previous segment about Competitors.
   - Emphasize that this is the final and most critical piece of the briefing.
   - Provide a brief outline of what will be covered in this segment (Stakeholder Identification, Key Executives, Stakeholder Contributions, and Relationship Strategies).

2. Stakeholder Identification:
   - Identify stakeholders frequently mentioned across initiatives:
     - Refer to Personas supporting these initiatives.
     - Explain why these stakeholders are crucial to multiple efforts and how they influence success.
   - Present this section as part of the overall narrative.

3. Key Executives:
   - Select 3-5 key executives from the identified stakeholders and for each:
     - Highlight the specific initiatives they are focused on.
     - Provide a brief bio from the key contacts summary section, including:
       * Their role and responsibilities.
       * Notable achievements or areas of expertise.
       * How their background aligns with the initiatives they support.
   - Integrate this section smoothly into the talk.

4. Stakeholder Contributions:
   - Draw connections between stakeholders’ expertise and their contributions to initiatives:
     - Discuss how their unique skills drive success.
     - Highlight potential synergies between different stakeholders' efforts.
   - Present this section as a continuation of the key executives discussion.

5. Relationship Strategies:
   - Discuss strategies for building strong relationships with these key stakeholders:
     - Suggest practical engagement methods tailored to each stakeholder’s priorities.
     - Emphasize how strong relationships can directly impact initiative outcomes.
   - Integrate this section naturally into the narrative.

6. Tone and Storytelling:
   - Incorporate relevant sales and marketing buzzwords naturally into the conversation while maintaining an upbeat tone.
   - Use storytelling techniques to make the discussion relatable and engaging, referencing real-world examples where appropriate.

7. Conclusion of Segment:
   - Summarize key takeaways about stakeholders’ roles in driving initiatives:
     - Reinforce the importance of understanding their motivations and aligning strategies accordingly.
     - Provide a motivational call-to-action for listeners, encouraging them to leverage these insights in their own strategic planning.

8. Talk Closing:
   - End with a reflective closing statement:
     - Thank listeners for joining this deep dive into strategic priorities.
     - Encourage them to apply what they’ve learned to build stronger partnerships and achieve their goals.

9. Dialogue Style:
    - Focus on natural conversational phrasing without naming the speaker.
    - Encourage lighthearted yet professional dialogue when appropriate.

10. TED Talk-Style Approach:
    - Structure this segment like a TED Talk by clearly outlining key points at the start, diving into each topic with engaging storytelling, and wrapping up with an inspiring conclusion.
    - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights.

By following these updated guidelines, craft a concise yet impactful TED Talk segment that concludes the series with actionable insights while leaving listeners motivated and inspired.

Script:
"""
        }
        return prompts[section].format(
            file_contents=content,
            account_name=account,
            client_name=client
        )

    def _create_abm_prompt(self, content: str, account: str, client: str) -> str:
        """
        Create GPT prompt for ABM reports
        Returns:
            Formatted prompt string
        """
        file_contents=content,
        account_name=account,
        client_name=client
        prompt = f"""
Create a dynamic and engaging TED Talk on "Next Quarter's ABM Deep Dive" focusing on strategically important personas for {account_name}, using the provided content as the foundation:

{file_contents}

Follow these detailed guidelines to ensure the talk is conversational, insightful, and actionable:

1. Introduction:
   - Start exactly with "Welcome to Next Quarter's ABM summary on {account_name}'s key personas, prepared exclusively for {client_name}."
   - Provide a concise outline of the talk's key areas: top ABM strategies, persona overview with themes and initiatives, and Account-Based Marketing channel strategies. Emphasize how understanding these elements can drive more effective engagement.

2. Top ABM Strategies:
   - Summarize the top 3 recommendations from the "Potential ABM Strategies Based on Intelligence" section.
   - For each strategy:
     - Provide a brief overview of the strategy.
     - Highlight key actions and targeted personas.
     - Discuss the potential impact and alignment with {client_name}'s offerings (approximately 50 words).

3. Persona Deep Dive:
   - Reference the "Persona Based Overlap of Themes and Initiatives" and "Channel Prediction for ABM to Identified Personas" sections.
   - Merge the information from these tables to create a comprehensive view of each persona.
   - Identify the top 4 personas based on strategic importance and relevance to {client_name}.
   - For each selected persona:
     - Persona Summary:
       - Clearly state the persona's name and job title. Provide a concise summary of their core responsibilities and overall influence within {account_name}.
     - Themes and Initiatives:
       - Summarize the key themes and initiatives associated with this persona, focusing on those most relevant to {client_name}'s offerings.
     - ABM Channels:
       - Detail the recommended ABM channels for engaging this persona. Explain how these channels align with the persona's preferences and information consumption habits.
   - Keep each persona discussion concise, while focusing on actionable insights.

4. Tone and Delivery:
   - Maintain an enthusiastic and conversational tone throughout the talk.
   - Avoid overly long pauses and forced transitions.
   - Use storytelling techniques to make the discussion relatable and engaging, referencing real-world examples where appropriate.

5. Structure:
   - Conclude with a summary of key takeaways, reinforcing the importance of understanding these personas, aligning with their initiatives, and leveraging appropriate ABM channels. Include a clear and motivational call to action, encouraging listeners to take concrete steps to engage these individuals.
   - End with: "You can always find more details about {account_name}'s key influencers in the full ABM report provided by Next Quarter."

6. Dialogue Style:
   - Focus on natural conversational phrasing without naming the speaker.
   - Encourage lighthearted yet professional dialogue when appropriate.

7. Strategic Focus (TED Talk Style):
   - Emulate a TED Talk by clearly outlining key points at the beginning, delving into each strategy and persona with engaging, story-driven insights, and concluding with an inspiring and actionable summary.
   - Use compelling examples and narratives to keep listeners engaged while conveying actionable insights about influencing these personas and driving meaningful engagement through appropriate ABM channels.

By following these guidelines, create a compelling TED Talk that is informative, motivating, and captivates listeners while effectively addressing strategic priorities. Focus on providing actionable advice for engaging key personas through targeted ABM strategies and channels to drive results for {client_name}.

Script:
""".format(
            file_contents=content,
            account_name=account,
            client_name=client
        )
        return prompt

    def _generate_script(self, prompt: str) -> str:
        """
        Generate script using OpenAI GPT-4
        Args:
            prompt: Complete prompt for script generation
        Returns:
            Cleaned script text
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional ted talk scriptwriter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return self._process_script(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Script generation failed: {str(e)}")
            raise

    def _process_script(self, script: str) -> str:
        """Clean up generated script formatting"""
        return '\n'.join([line.strip().replace('*', '') 
                        for line in script.split('\n') if line.strip()])

    def _text_to_speech(self, script: str) -> Optional[bytes]:
        """
        Convert text to audio bytes using OpenAI TTS
        Args:
            script: Text script to convert
        Returns:
            MP3 audio bytes
        """
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1-hd",
                voice="ash",
                input=script
            )
            return response.content
        except Exception as e:
            logger.error(f"TTS failed: {str(e)}")
            return None

# API Startup Configuration
@app.on_event("startup")
async def startup_event():
    """Initialize service components on application startup"""
    app.state.generator = BriefingGenerator({
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "source_bucket": os.getenv("S3_SOURCE_BUCKET"),
        "destination_bucket": os.getenv("S3_DEST_BUCKET"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1")
    })

# API Endpoints
@app.post("/process-file/", status_code=status.HTTP_202_ACCEPTED)
async def process_file(request: ProcessingRequest):
    """
    Main processing endpoint
    Example Request Body:
    {
        "s3_key": "reports/2024/q3/Global_Camping_Suppliers_Intelligence_Report_For_National_Outdoor_Retailers.pdf",
        "report_type": "intelligence_report"
    }
    """
    try:
        result = app.state.generator.process_file(request.s3_key)
        return result if result["status"] == "success" else HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )

# Application Entry Point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile=os.getenv("SSL_KEYFILE"),
        ssl_certfile=os.getenv("SSL_CERTFILE")
    )