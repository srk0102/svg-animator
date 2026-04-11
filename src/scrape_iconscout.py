"""IconScout Lottie scraper — Playwright automation.

Strategy:
  1. Loop through configured categories (business, character, people, etc.)
  2. For each category, loop pages 1..N
  3. On each listing page, collect all individual animation detail-page URLs
  4. Visit each detail page, intercept the CloudFront Lottie JSON that
     lottie-web fetches to render the preview
  5. Save raw JSON + metadata to data/private_scraped/iconscout/
  6. Dedupe by numeric ID, resume from scraper_state.json

To stay under the radar:
  - Uses Playwright persistent_context pointing at a real Chrome user data
    dir so your login + cookies + fingerprint are reused
  - Random human-like delays between page loads
  - Only visits pages that normal users would visit

PRIVATE DATA — never commit, never publish. gitignored.

Usage:
  python src/scrape_iconscout.py --test                 # 1 category, 1 page
  python src/scrape_iconscout.py --category business    # single category
  python src/scrape_iconscout.py                        # full run
"""
import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

OUTPUT_DIR = Path("data/private_scraped/iconscout")
STATE_FILE = Path("scraper_state.json")

CATEGORIES = [
    "character",
    "people",
    "emoji",
    "gesture",
    "emotion",
    "education",
    "medical",
    "technology",
    "ecommerce",
    "food",
    "animal",
    "sport",
    "nature",
    "travel",
]

LOTTIE_URL_RE = re.compile(r"cloudfront\.net/lottie/.*\.json", re.IGNORECASE)
ID_RE = re.compile(r"/(\d+)\.json")

MAX_PAGES_PER_CATEGORY = 50
MAX_DETAILS_PER_PAGE = 100
LISTING_SETTLE_MS = 2500
DETAIL_SETTLE_MS = 3500


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"scraped_ids": [], "categories_done": {}}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def human_delay(lo=0.8, hi=2.2):
    return random.uniform(lo, hi)


async def safe_eval(page, expr, retries=3):
    """Evaluate JS with retry on 'execution context destroyed' errors."""
    for attempt in range(retries):
        try:
            return await page.evaluate(expr)
        except Exception as e:
            if "context was destroyed" in str(e).lower() or "navigation" in str(e).lower():
                await page.wait_for_timeout(1500)
                continue
            raise
    return None


async def collect_detail_links(page, category, page_num):
    """On a category listing page, find all individual animation detail URLs."""
    listing_url = f"https://iconscout.com/lottie-animations/{category}?page={page_num}"
    try:
        await page.goto(listing_url, timeout=45000, wait_until="domcontentloaded")
        # Wait for any client-side redirects / SPA navigation to settle
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
    except Exception as e:
        print(f"    Failed to load listing: {e}")
        return []

    await page.wait_for_timeout(LISTING_SETTLE_MS)

    # Scroll to trigger lazy renders of item tiles (safe_eval handles SPA races)
    for _ in range(4):
        await safe_eval(page, "window.scrollBy(0, window.innerHeight)")
        await page.wait_for_timeout(800)
    await safe_eval(page, "window.scrollTo(0, 0)")
    await page.wait_for_timeout(600)

    # Extract animation detail-page links + debug dump of all iconscout hrefs
    result = await safe_eval(page, """
        () => {
            const anchors = Array.from(document.querySelectorAll('a[href]'));
            const all_iconscout = [];
            const seen = new Set();
            for (const a of anchors) {
                const href = a.href;
                if (!href || !href.includes('iconscout.com')) continue;
                if (seen.has(href)) continue;
                seen.add(href);
                all_iconscout.push(href);
            }
            // Filter candidates: any URL with a trailing -<digits> or _<digits>
            // that likely represents an individual animation (not collections)
            const candidates = all_iconscout.filter(h =>
                /[-_]\\d{4,}(?:[\\/?#]|$)/.test(h) &&
                !h.includes('/collection') &&
                !h.includes('/pack') &&
                !h.includes('/contributors')
            );
            return { all: all_iconscout, candidates: candidates };
        }
    """)
    all_links = result["all"]
    candidates = result["candidates"]
    # Debug: show a sample so we can fix the pattern if needed
    print(f"    [debug] total iconscout.com links: {len(all_links)}")
    if len(candidates) == 0 and len(all_links) > 0:
        print(f"    [debug] no candidates matched. Sample links:")
        for h in all_links[:15]:
            print(f"      {h}")
    return candidates


async def scrape_detail_page(page, detail_url, captured, state):
    """Visit detail page, capture any CloudFront Lottie JSONs that load."""
    try:
        await page.goto(detail_url, timeout=30000, wait_until="domcontentloaded")
    except Exception as e:
        print(f"      goto error: {e}")
        return 0

    # Wait for lottie-web to fetch the JSON
    await page.wait_for_timeout(DETAIL_SETTLE_MS)
    # Trigger any hover/interaction that might prompt load
    try:
        await page.evaluate("window.scrollBy(0, 400)")
    except Exception:
        pass
    await page.wait_for_timeout(800)

    # Count newly captured in this visit
    return len(captured)


async def run_category(context, category, state, max_pages, visit_details=False):
    page = await context.new_page()

    # ---- Network interception: capture any response matching Lottie CDN pattern ----
    captured_in_session = []
    scraped_ids = set(state["scraped_ids"])
    cat_dir = OUTPUT_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    async def on_response(resp):
        url = resp.url
        if not LOTTIE_URL_RE.search(url):
            return
        try:
            body = await resp.body()
            if len(body) < 200 or len(body) > 20_000_000:
                return
            try:
                data = json.loads(body)
            except Exception:
                return
            if not isinstance(data, dict) or "layers" not in data or "ip" not in data:
                return

            # Extract ID
            m = ID_RE.search(url)
            lid = m.group(1) if m else f"unknown_{int(time.time()*1000)}"
            if lid in scraped_ids:
                return
            scraped_ids.add(lid)

            out_path = cat_dir / f"{lid}.json"
            out_path.write_bytes(body)
            captured_in_session.append(lid)

            meta = {
                "id": lid,
                "category": category,
                "source_url": url,
                "v": data.get("v"),
                "w": data.get("w"),
                "h": data.get("h"),
                "fr": data.get("fr"),
                "op": data.get("op"),
                "layer_count": len(data.get("layers", [])),
                "captured_at": time.time(),
            }
            (cat_dir / f"{lid}.meta.json").write_text(json.dumps(meta))

            if len(captured_in_session) % 10 == 0:
                print(f"    [{category}] captured {len(captured_in_session)} new "
                      f"(total unique: {len(scraped_ids)})")
        except Exception as e:
            # "No resource with given identifier found" = Chrome freed the response body
            # before we could read it. Happens on SPA navigation, harmless.
            if "No resource with given identifier" not in str(e):
                print(f"    intercept error: {e}")

    page.on("response", on_response)

    start_page = state["categories_done"].get(category, 0) + 1
    consecutive_empty = 0
    for page_num in range(start_page, max_pages + 1):
        print(f"\n  [{category}] listing page {page_num}")
        before_listing = len(captured_in_session)
        links = await collect_detail_links(page, category, page_num)
        # Scrolling the listing page already triggers Lottie loads — give it
        # a final beat so the last batch of responses can settle
        await page.wait_for_timeout(1500)
        after_listing = len(captured_in_session)
        listing_gain = after_listing - before_listing
        print(f"    listing scroll captured {listing_gain} (found {len(links)} detail links)")

        if visit_details and links:
            for detail_url in links[:MAX_DETAILS_PER_PAGE]:
                await scrape_detail_page(page, detail_url, captured_in_session, state)
                await asyncio.sleep(human_delay(0.3, 0.9))

        # Detect end-of-results: IconScout serves stale items from the last
        # page when you query past the end, so we can't rely on len(links)==0.
        # Instead, stop when 2 consecutive pages yield 0 NEW captures (all dupes).
        if listing_gain == 0:
            consecutive_empty += 1
            print(f"    (0 new — likely past last page, {consecutive_empty}/2)")
            if consecutive_empty >= 2:
                print(f"    no more new results in {category}, stopping")
                break
        else:
            consecutive_empty = 0

        # Checkpoint after each listing page
        state["categories_done"][category] = page_num
        state["scraped_ids"] = list(scraped_ids)
        save_state(state)

        # Delay between listing pages (human-ish)
        await asyncio.sleep(human_delay(1.5, 3.0))

    print(f"\n  [{category}] DONE. session: {len(captured_in_session)}, "
          f"total unique: {len(scraped_ids)}")
    await page.close()
    return len(captured_in_session)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="1 category, 1 page")
    parser.add_argument("--category", help="scrape only this category")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_CATEGORY)
    parser.add_argument("--user-data-dir", default=str(Path.home() / ".iconscout_chrome"),
                        help="Chrome profile dir (persistent, re-used across runs)")
    parser.add_argument("--headed", action="store_true", help="show browser (helpful for login)")
    parser.add_argument("--skip-login-check", action="store_true",
                        help="skip the manual login prompt (use if already logged in)")
    parser.add_argument("--chrome-profile", default=None,
                        help="use an existing Chrome profile by name (e.g., 'srk'). "
                             "Close Chrome before running. Overrides --user-data-dir.")
    parser.add_argument("--visit-details", action="store_true",
                        help="also visit individual detail pages (slow). By default, "
                             "we only scroll listing pages since they already load the JSONs.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    # Resolve --chrome-profile: COPY the profile to a dedicated scraper dir.
    # This avoids lock conflicts with your real Chrome and keeps it untouched.
    chrome_profile_dir = None
    if args.chrome_profile:
        import shutil
        chrome_user_data = Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
        local_state_file = chrome_user_data / "Local State"
        if not local_state_file.exists():
            print(f"  ERROR: Chrome user data dir not found at {chrome_user_data}")
            return
        try:
            ls = json.loads(local_state_file.read_text(encoding="utf-8"))
            profiles = ls.get("profile", {}).get("info_cache", {})
        except Exception as e:
            print(f"  ERROR: could not read Local State: {e}")
            return
        target = args.chrome_profile.lower()
        match = None
        for folder, info in profiles.items():
            name = info.get("name", "").lower()
            gaia = info.get("gaia_name", "").lower()
            user = info.get("user_name", "").lower()
            if target in (name, gaia, user) or target in name:
                match = folder
                break
        if not match:
            print(f"  ERROR: profile '{args.chrome_profile}' not found. Available:")
            for folder, info in profiles.items():
                print(f"    {folder}: name={info.get('name')!r} user={info.get('user_name')!r}")
            return

        source_profile = chrome_user_data / match
        dest_root = Path(args.user_data_dir)
        dest_profile = dest_root / "Default"

        if not dest_profile.exists():
            print(f"  First-time setup: copying profile '{args.chrome_profile}' "
                  f"({match}) -> {dest_profile}")
            print(f"  This takes ~10-30 seconds. Only happens once.")
            dest_root.mkdir(parents=True, exist_ok=True)

            # Copy only the files that matter for session/cookies.
            # Skip huge caches, extensions, history etc.
            keep_files = [
                "Cookies", "Cookies-journal",
                "Login Data", "Login Data-journal",
                "Preferences", "Secure Preferences",
                "Web Data", "Web Data-journal",
                "History", "History-journal",
            ]
            keep_dirs = [
                "Local Storage", "Session Storage", "IndexedDB",
                "Service Worker",
            ]

            dest_profile.mkdir(parents=True, exist_ok=True)
            for f in keep_files:
                src = source_profile / f
                if src.exists():
                    try:
                        shutil.copy2(src, dest_profile / f)
                    except Exception as e:
                        print(f"    warn: copy {f}: {e}")
            for d in keep_dirs:
                src = source_profile / d
                if src.exists():
                    try:
                        shutil.copytree(src, dest_profile / d, dirs_exist_ok=True)
                    except Exception as e:
                        print(f"    warn: copytree {d}: {e}")

            # Also copy Local State (needed at user_data_dir level)
            src_ls = chrome_user_data / "Local State"
            if src_ls.exists():
                try:
                    shutil.copy2(src_ls, dest_root / "Local State")
                except Exception as e:
                    print(f"    warn: copy Local State: {e}")

            print(f"  Profile copied successfully.")
        else:
            print(f"  Using existing profile copy at: {dest_profile}")

        chrome_profile_dir = "Default"
        print(f"  Source: {source_profile}")
        print(f"  Scraper profile: {dest_profile}")

    if args.test:
        categories = ["business"]
        max_pages = 1
    elif args.category:
        categories = [args.category]
        max_pages = args.max_pages
    else:
        categories = CATEGORIES
        max_pages = args.max_pages

    print("=" * 60)
    print("IconScout Lottie Scraper")
    print(f"  Categories: {categories}")
    print(f"  Max pages/cat: {max_pages}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Chrome profile: {args.user_data_dir}")
    print(f"  Already scraped: {len(state['scraped_ids'])} unique IDs")
    print("=" * 60)

    async with async_playwright() as p:
        launch_args = ["--disable-blink-features=AutomationControlled"]
        if chrome_profile_dir:
            launch_args.append(f"--profile-directory={chrome_profile_dir}")

        launch_kwargs = dict(
            user_data_dir=args.user_data_dir,
            headless=not args.headed,
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            args=launch_args,
        )
        # When using a real Chrome profile, use installed Chrome (channel=chrome)
        # instead of Playwright's bundled Chromium for compatibility
        if chrome_profile_dir:
            launch_kwargs["channel"] = "chrome"

        context = await p.chromium.launch_persistent_context(**launch_kwargs)

        # ---- Login gate: open IconScout and wait for user confirmation ----
        if not args.skip_login_check:
            login_page = await context.new_page()
            try:
                await login_page.goto("https://iconscout.com/lottie-animations",
                                      timeout=45000, wait_until="domcontentloaded")
                await login_page.wait_for_timeout(2000)
            except Exception as e:
                print(f"  (warning) couldn't pre-load homepage: {e}")

            print()
            print("=" * 60)
            print("  LOGIN CHECK")
            print("=" * 60)
            print("  1. A browser window should be open on iconscout.com")
            print("  2. If you see a 'Login' button top-right, click it and log in")
            print("  3. If you're already logged in (avatar in top-right), you're set")
            print("  4. Navigate around if you want — cookies will persist")
            print()
            try:
                ans = input("  Are you logged in and ready to start scraping? [y/N]: ").strip().lower()
            except EOFError:
                ans = "n"
            if ans != "y":
                print("  Aborting. Log in next time and try again.")
                await login_page.close()
                await context.close()
                return
            await login_page.close()
            print("  Login confirmed. Starting scrape...\n")

        start = time.time()
        total = 0
        try:
            for cat in categories:
                try:
                    total += await run_category(context, cat, state, max_pages,
                                                visit_details=args.visit_details)
                except KeyboardInterrupt:
                    print("\n\nInterrupted — state saved.")
                    break
                except Exception as e:
                    print(f"\n  ERROR in {cat}: {e}")
        finally:
            await context.close()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Session total: {total} new files in {elapsed/60:.1f} min")
    print(f"Unique files on disk: {len(state['scraped_ids'])}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
