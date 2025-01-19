from loguru import logger
from pymongo import MongoClient
from clearml import PipelineController, Task
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import requests


# Combined Scrape and Store Task
def scrape_and_store_task(links: list[str]):
    """
    Scrape content from the provided links and store raw data in MongoDB.
    """
    def identify_platform(link: str) -> str:
        if "github.com" in link:
            return "github"
        elif "medium.com" in link:
            return "medium"
        elif "youtube.com" in link or "youtu.be" in link:
            return "youtube"
        else:
            return "other"

    def scrape_github(link: str) -> dict:
        """
        Scrape GitHub repository details and extract code file contents.
        """
        def fetch_github_files(api_url: str, headers: dict) -> str:
            """
            Fetch files recursively from a GitHub repository.
            """
            try:
                response = requests.get(api_url, headers=headers, timeout=10)
                response.raise_for_status()
                contents = response.json()

                file_contents = []
                for item in contents:
                    if item["type"] == "file":
                        file_response = requests.get(item["download_url"], timeout=10)
                        file_response.raise_for_status()
                        file_contents.append(f"File: {item['name']}\n{file_response.text}")
                    elif item["type"] == "dir":
                        subdir_files = fetch_github_files(item["url"], headers)
                        file_contents.extend(subdir_files)

                return "\n".join(file_contents)
            except Exception as e:
                logger.error(f"Failed to fetch files from {api_url}: {e}")
                return ""

        try:
            # Extract owner and repo name from the URL
            parts = link.strip("/").split("/")
            if len(parts) < 5:
                raise ValueError("Invalid GitHub URL format.")
            owner, repo = parts[3], parts[4]

            # GitHub API URLs
            contents_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token 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"
            }

            # Fetch Repository Files Recursively
            file_contents = fetch_github_files(contents_api_url, headers)

            return {"platform": "github", "content": file_contents, "link": link}
        except Exception as e:
            logger.error(f"GitHub scrape failed for {link}: {e}")
            return {"platform": "github", "content": None, "link": link, "error": str(e)}

    def scrape_medium(link: str) -> dict:
        try:
            response = requests.get(link, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No Title"
            content = soup.find("article").get_text(separator="\n").strip() if soup.find("article") else "No Content"
            return {"platform": "medium", "content": f"Title: {title}\n\n{content}", "link": link}
        except Exception as e:
            logger.error(f"Medium scrape failed for {link}: {e}")
            return {"platform": "medium", "content": None, "link": link, "error": str(e)}

    def scrape_youtube(link: str) -> dict:
        try:
            video_id = link.split("v=")[-1] if "v=" in link else link.split("/")[-1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = "\n".join([entry["text"] for entry in transcript])
            return {"platform": "youtube", "content": transcript_text, "link": link}
        except Exception as e:
            logger.error(f"YouTube scrape failed for {link}: {e}")
            return {"platform": "youtube", "content": None, "link": link, "error": str(e)}

    def scrape_other(link: str) -> dict:
        """
        Fallback scraper for other platforms.
        """
        try:
            response = requests.get(link, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.title.string if soup.title else "Untitled"
            text_content = soup.get_text(separator="\n").strip()
            return {"platform": "other", "content": f"Title: {title}\n\n{text_content}", "link": link}
        except Exception as e:
            logger.error(f"Fallback scrape failed for {link}: {e}")
            return {"platform": "other", "content": None, "link": link, "error": str(e)}

    scraped_data = []
    for link in links:
        platform = identify_platform(link)
        if platform == "github":
            scraped_data.append(scrape_github(link))
        elif platform == "medium":
            scraped_data.append(scrape_medium(link))
        elif platform == "youtube":
            scraped_data.append(scrape_youtube(link))
        else:
            scraped_data.append(scrape_other(link))

    logger.info(f"Raw Scraped Data: {scraped_data}")  # Debugging log

    def connect_to_mongo():
        # MongoDB Configuration
        MONGO_URI = "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
        DB_NAME = "etl_database_n"
        COLLECTION_NAME = "platform_scraped_data"
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        return collection

    # Store data in MongoDB
    collection = connect_to_mongo()
    for record in scraped_data:
        try:
            if record.get("content"):  # Only store records with valid content
                collection.insert_one(record)
                logger.info(f"Inserted record for link: {record.get('link')}")
        except Exception as e:
            logger.error(f"Failed to insert record into MongoDB: {e}")

# ClearML Pipeline
def run_pipeline():
    # Initialize ClearML Task
    task = Task.init(project_name="Web Scraping Pipeline", task_name="Local Scraping and Storing Pipeline")

    # Define Links to Scrape
    links = [
        "https://abdulrazzaq0902.medium.com/introduction-to-ros2-robot-operating-system-9fa9e9367263",
        "https://medium.com/@arshad.mehmood/enabling-multi-robot-arm-in-gazebo-for-ros2-dc18981c03c6",
        "https://automaticaddison.com/how-to-configure-moveit-2-for-a-simulated-robot-arm/",
        "https://medium.com/@arshad.mehmood/enabling-multi-robot-arm-in-gazebo-for-ros2-dc18981c03c6",
        "https://medium.com/@santoshbalajimanipulation-with-moveit2-visualizing-robot-arm-in-simulation-1-8cd3a46d42b4",
        "https://medium.com/robotics-zone/ros2-navigation-tutorial-and-implementation-guide-6c6e362dbf70",
        "https://medium.com/robotics-zone/understanding-moveit2-and-ros-2-for-robotics-61d74832cd2a",
        "https://medium.com/ros2-basics/building-your-own-ros2-navigation-stack-from-scratch-cc9c6b6a32e6",
        "https://medium.com/robotics-zone/introducing-ros2-and-gazebo-simulation-for-robotics-3e4054d4f8f8",
        "https://medium.com/robotics-zone/using-ros2-and-gazebo-to-simulate-robots-in-a-vibrant-world-34a6a4f35b28",
        "https://automaticaddison.com/how-to-configure-moveit-2-for-a-simulated-robot-arm/",
        "https://automaticaddison.com/complete-guide-to-the-moveit-setup-assistant-for-moveit-2/",
        "https://www.reddit.com/r/ROS/comments/1448sd2/how_to_inegrate_ros2_humble_with_gazebo_and/",
        "https://medium.com/@arshad.mehmood/enabling-multi-robot-arm-in-gazebo-for-ros2-dc18981c03c6",
        "https://medium.com/@santoshbalajimanipulation-with-moveit2-visualizing-robot-arm-in-simulation-1-8cd3a46d42b4",
        "https://medium.com/robotics-zone/ros2-navigation-tutorial-and-implementation-guide-6c6e362dbf70",
        "https://medium.com/robotics-zone/understanding-moveit2-and-ros-2-for-robotics-61d74832cd2a",
        "https://medium.com/ros2-basics/building-your-own-ros2-navigation-stack-from-scratch-cc9c6b6a32e6",
        "https://medium.com/robotics-zone/introducing-ros2-and-gazebo-simulation-for-robotics-3e4054d4f8f8",
        "https://medium.com/robotics-zone/using-ros2-and-gazebo-to-simulate-robots-in-a-vibrant-world-34a6a4f35b28",
        "https://www.youtube.com/watch?v=kR7w5uvykRg",
        "https://www.youtube.com/watch?v=KAyX4Mf5n2E",
   
    ]

    # Define Pipeline
    pipe = PipelineController(
        name="Scraping and Storing Pipeline",
        project="ETL Pipeline",
        version="1.0"
    )

    # Single Step: Scrape and Store
    pipe.add_function_step(
        name="Scrape and Store",
        function=scrape_and_store_task,
        function_kwargs={"links": links}
    )

    # Execute Pipeline
    pipe.start_locally(run_pipeline_steps_locally=True)
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
