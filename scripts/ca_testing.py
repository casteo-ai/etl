from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

PROFILE = "/home/poucho/code/etl/browser_profiles/cryptoarchive-firefox"

with sync_playwright() as p:
    print("Launching Firefox...")
    context = p.firefox.launch_persistent_context(
        user_data_dir=PROFILE,
        executable_path="/usr/bin/firefox",
        headless=False,
        accept_downloads=True,
    )

    page = context.pages[0] if context.pages else context.new_page()
    page.set_default_timeout(15000)

    print("Opening site...")
    try:
        page.goto("https://www.cryptoarchive.com.au/signup", wait_until="domcontentloaded", timeout=30000)
        print("Loaded:", page.url)
    except PlaywrightTimeoutError as exc:
        print("Navigation timed out:", exc)

    print("Title:", page.title())
    input("If the page is blank, try typing the URL manually in the browser. Press Enter here when done...")

    print("Current URL:", page.url)
    print("Title:", page.title())
    context.close()
