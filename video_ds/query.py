from selenium import webdriver
import time
from bs4 import BeautifulSoup
import json

########### open browser ############
# driver = webdriver.Chrome(executable_path="./chromedriver")
# driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

driver = webdriver.Edge(executable_path="./msedgedriver")
driver.get("https://www.youtube.com/")
driver.maximize_window()
time.sleep(1)
driver.refresh()

cookie = driver.get_cookies()


def get_videos(video_num):
    video_ids = list()
    stop_flag = False
    while True:
        html = driver.page_source
        page = BeautifulSoup(html, 'html.parser')
        videos = page.find_all('a', id="thumbnail")

        for item in videos:
            video = item.get("href")
            if video is not None and "/watch?v=" in video:
                video_id = video.replace('/watch?v=', '')
                if video_id not in video_ids:
                    video_ids.append(video_id)
                print(video_id)
                if len(video_ids) > video_num:
                    stop_flag = True
                    break
        if stop_flag:
            break
        js = "var q=document.documentElement.scrollTop=100000000000"
        driver.execute_script(js)
        time.sleep(3)  # 等待页面刷新
    return video_ids


keywords = ["natural | nature sounds | landscape and river"]
final_res = dict()
for query in keywords:
    # query video by keywords and condition with 4-20mins/4K/HD/license
    url = 'https://www.youtube.com/results?search_query=' + query + '&sp=EggYAyABMAFwAQ%253D%253D'
    driver.get(url)
    print(query)

    video_ids = get_videos(400)
    final_res[query] = video_ids
    time.sleep(1)

json_obj = json.dumps(final_res, indent=4)
with open("search_res_400.json", "w") as outfile:
    outfile.write(json_obj)

# exist broswer
driver.quit()
