import os, sys
import time
import pandas as pd
import shutil

from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
options = Options()
#init 
options.add_argument('--disable-gpu')
options.add_argument('--disable-extensions')
options.add_argument('--proxy-server="direct://"')
options.add_argument('--proxy-bypass-list=*')
options.add_argument('--start-maximized')
import subprocess
# import warnings 
# warnings.simple

"""
参照記事
〇webdriver(chrome)の使い方
https://tanuhack.com/selenium/#WebDriver
〇firefoxで自動ダウンロード(確認画面を消す)
  https://qiita.com/youngsend/items/25ac8ea6c176182db8f4
〇複数タブで同時に開くときにポップアップをブロックされる問題を解決する
  https://www.soudegesu.com/post/python/selenium-firefox-tab-restriction/
〇子を要素として、親の要素をを見つける
  https://qiita.com/ShortArrow/items/ced7bf23b806c0d835da
〇複数クラスのclassに関してを抽出する記述
https://qiita.com/hanonaibaobabu/items/e547410865d857aa25ec
"""

class Img:
  def __init__(self,datadir):
    #firefox setting
    #---driverpath---------
    driver_path="C:\\Users\\1119041\\Desktop\\geckodriver-v0.27.0-win64\\chromedriver.exe"
    # self.driver = webdriver.Firefox(firefox_profile=fp, executable_path=driver_path,options=options)
    # self.driver.implicitly_wait(10)
    self.driver = webdriver.Chrome(executable_path=driver_path, chrome_options=options)
    self.driver.implicitly_wait(10) # 秒
    # sys.exit()
    #chrome の設定方法
# ファイルのデフォルトの保存先を変更する
    self.driver.command_executor._commands['send_command'] = (
        'POST',
        '/session/$sessionId/chromium/send_command'
    )
    params = {
        'cmd': 'Page.setDownloadBehavior', 
        'params': {
            'behavior': 'allow',
            'downloadPath': datadir
        }
    }
    self.driver.execute('send_command', params=params)
    print(" [end] Initialize ....")
    
  def get(self,query="渡邉理佐"):
    self.driver.get("https://www.google.com")
    input_area = self.driver.find_element_by_name("q")
    # search_bar.send_keys(input("何の画像を検索しますか？:"))
    input_area.send_keys(query)
    input_area.submit()
    sys.exit()
    time.sleep(3)
    self.driver.quit()
    # time.sleep(3)
    return


if __name__ == "__main__":
  #init
  datadir="C:\\Users\\1119041\\Desktop\\sori_workspace\\selenium\\dat2\\img"
  subprocess.run("rm *.png" ,shell=True,cwd = datadir)
  subprocess.run("rm *.jpg" ,shell=True,cwd = datadir)
  query_text ="渡邉理佐"
  driver = Img(datadir=datadir) #instance
  driver.get(query = query_text)
  sys.exit()