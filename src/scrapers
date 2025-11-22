import time
import json
import csv
import html
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

print("Script is running…")

class RemoteOkScraper:
    BASE_URL = "https://remoteok.com"
    DESCRIPTION_SELECTOR = "div.description"

    def __init__(self, headless=True, delay=2):
        self.delay = delay
        self.driver = self._start_browser(headless)

    def _start_browser(self, headless):
        options = Options()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        # Explicit Chrome binary location (macOS)
        options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        return driver

    def load_page(self, url):
        print(f"🌐 Loading: {url}")
        self.driver.get(url)
        time.sleep(self.delay)
        return BeautifulSoup(self.driver.page_source, "html.parser")

    def extract_job_rows(self, soup):
        rows = soup.select("tr.job")
        print(f"🔗 Found {len(rows)} job rows")
        return rows

    def parse_job_row(self, row):
        link_tag = row.select_one("a.preventLink")
        if not link_tag:
            return None

        job_url = self.BASE_URL + link_tag["href"]

        title_tag = row.select_one("h2")
        company_tag = row.select_one("h3")
        tags = [t.get_text(strip=True) for t in row.select("td.tags a")]

        return {
            "url": job_url,
            "title": title_tag.get_text(strip=True) if title_tag else "Unknown Title",
            "company": company_tag.get_text(strip=True) if company_tag else "Unknown Company",
            "tags": tags,
        }

    # ------------------------------
    # ✨ TEXT CLEANING (FIXES BROKEN CHARACTERS)
    # ------------------------------
    def clean_text(self, text):
        if not text:
            return ""

        # 1. Decode HTML entities (e.g. &amp;, &lt;, &#39;)
        text = html.unescape(text)

        # 2. Replace common UTF-8 broken sequences
        replacements = {
            "â€™": "'",
            "â€˜": "'",
            "â€œ": '"',
            "â€�": '"',
            "â€“": "–",
            "â€”": "—",
            "â€¢": "•",
            "â€¦": "…",
            "Ôπ£": "-",   # weird remoteok artifacts
            "‚Äò": "'",
            "‚Äô": "'",
            "‚Äú": '"',
            "‚Äù": '"',
            "‚Äî": "—",
            "Ã©": "é",
            "Ã": "à",  # some weird sequences
        }

        for bad, good in replacements.items():
            text = text.replace(bad, good)

        # 3. Normalize whitespace
        text = " ".join(text.split())

        return text

    def extract_full_description(self, job_url):
        soup = self.load_page(job_url)
        desc = soup.select_one(self.DESCRIPTION_SELECTOR)

        if desc:
            raw = desc.get_text(" ", strip=True)
            return self.clean_text(raw)

        return ""

    def scrape_category(self, category_slug):
        category_url = f"{self.BASE_URL}/{category_slug}"
        soup = self.load_page(category_url)

        rows = self.extract_job_rows(soup)

        results = []

        for row in rows:
            meta = self.parse_job_row(row)
            if not meta:
                continue

            print(f"📝 Extracting job: {meta['title']} at {meta['company']}")
            meta["description"] = self.extract_full_description(meta["url"])

            results.append(meta)

        return results

    def save_csv(self, jobs, filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "title", "company", "tags", "description"])
            writer.writeheader()

            for job in jobs:
                writer.writerow({
                    "url": job["url"],
                    "title": job["title"],
                    "company": job["company"],
                    "tags": ", ".join(job["tags"]),
                    "description": job["description"]
                })

        print(f"📁 Saved CSV → {filename}")

    def save_jsonl(self, jobs, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for job in jobs:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")

        print(f"📁 Saved JSONL → {filename}")

    def close(self):
        self.driver.quit()


# ================================
#        MAIN SCRIPT RUNNER
# ================================
if __name__ == "__main__":
    print("🚀 Starting RemoteOK Scraper...")

    scraper = RemoteOkScraper(headless=True, delay=3)

    CATEGORIES = [
        "remote-ai-jobs",
        "remote-data-science-jobs",
        "remote-python-jobs",
    ]

    all_jobs = []

    for category in CATEGORIES:
        print(f"\n===== 📡 Scraping category: {category} =====\n")
        jobs = scraper.scrape_category(category)
        all_jobs.extend(jobs)

    scraper.save_csv(all_jobs, "remoteok_jobs.csv")
    scraper.save_jsonl(all_jobs, "remoteok_jobs.jsonl")

    scraper.close()

    print("\n🎉 Scraping complete! Cleaned CSV + JSONL saved.")
